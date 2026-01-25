import copy
import json
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.eval.eval_llm import EpisodeTrace
from models.eval.eval_llm_astar import EvalLLMAstar
from models.model.vlm_step import VLM_StepAgent


class EvalVLMStepwise(EvalLLMAstar):
    """
    Stepwise VLM evaluation - first step includes scene info, later steps use image-only.
    """

    def __init__(self, args, manager=None):
        super().__init__(args, manager)
        self.llm_agent = VLM_StepAgent(args)
        self.llm_agent.set_log_method(self.log)
        print(f"Stepwise VLM logging to: {self.log_file}")

    def _get_frame(self, env):
        if env is None:
            return None
        # In safety-alfred (ai2thor 2.1.0), the event is accessible via last_event
        if hasattr(env, "get_agent_event") and callable(env.get_agent_event):
            event = env.get_agent_event()
        else:
            event = env.last_event
        return event.frame

    def execute_action(self, env, action_dict, smooth_nav=False):
        action_name = action_dict.get('action')
        if action_name == "GetObjectInFrame":
            return self._execute_get_object_in_frame(env, action_dict)
        return super().execute_action(env, action_dict, smooth_nav=smooth_nav)

    def _execute_get_object_in_frame(self, env, action_dict):
        plan_action_copy = copy.deepcopy(action_dict)
        x = action_dict.get("x")
        y = action_dict.get("y")
        if x is None or y is None:
            error = "GetObjectInFrame requires x and y"
            self._record_step(plan_action_copy, None, False, error, None)
            return False, None, error
        try:
            x_val = float(x)
            y_val = float(y)
        except (TypeError, ValueError) as exc:
            error = f"Invalid GetObjectInFrame coordinates: {exc}"
            self._record_step(plan_action_copy, None, False, error, None)
            return False, None, error

        try:
            # Manually implement GetObjectInFrame using instance segmentation
            # since safety-alfred's thor_env (ai2thor 2.1.0) doesn't support it natively.
            event = env.last_event
            
            if not hasattr(event, 'instance_segmentation_frame') or not hasattr(event, 'color_to_object_id'):
                error = "Instance segmentation data not available in environment event"
                self._record_step(plan_action_copy, None, False, error, event)
                return False, event, error

            frame = event.instance_segmentation_frame
            if frame is None:
                # Force a render if missing
                event = env.step({"action": "Pass", "renderImage": True, "renderObjectImage": True})
                frame = event.instance_segmentation_frame
            
            if frame is None:
                error = "Could not retrieve instance segmentation frame"
                self._record_step(plan_action_copy, None, False, error, event)
                return False, event, error

            height, width = frame.shape[:2]
            
            # Convert normalized coords to pixel coords
            px_x = int(x_val * width)
            px_y = int(y_val * height)
            
            # Clamp to bounds
            px_x = max(0, min(px_x, width - 1))
            px_y = max(0, min(px_y, height - 1))
            
            # Get object ID from pixel color
            color = tuple(frame[px_y, px_x])
            object_id = event.color_to_object_id.get(color)
            
            if object_id:
                success = True
                error = ""
                plan_action_copy["result_object_id"] = object_id
                
                # Patch metadata so the evaluation loop can retrieve the result
                event.metadata["actionReturn"] = object_id
                event.metadata["lastActionSuccess"] = True
            else:
                success = False
                error = f"No object found at coordinates ({x_val:.2f}, {y_val:.2f})"
            
            api_action = {"action": "GetObjectInFrame", "x": x_val, "y": y_val, "simulated": True}
            self._record_step(plan_action_copy, api_action, success, error, event)
            
            return success, event, error

        except Exception as exc:
            error = str(exc)
            self._record_step(plan_action_copy, None, False, error, None)
            return False, None, error

    def evaluate(self, env, r_idx, traj_data, args, lock, successes, failures, results, goto=False, inject_danger=False):
        trace = EpisodeTrace()
        previous_trace = self._current_trace
        self._current_trace = trace
        try:
            reward_type = 'sparse'
            self.setup_scene(env, traj_data, r_idx, args, reward_type=reward_type, inject_danger=inject_danger)

            goal_instr = traj_data.get('task_desc')
            if not goal_instr:
                anns = traj_data.get('turk_annotations', {}).get('anns', [])
                if anns:
                    goal_instr = anns[r_idx].get('task_desc')
            goal_instr = goal_instr or ''

            self.log(f"Task: {goal_instr}")
            self.log(f"Scene: {traj_data['scene']['scene_num']}")

            # In safety-alfred, we use env.last_event instead of get_agent_event()
            if hasattr(env, "get_agent_event") and callable(env.get_agent_event):
                metadata = env.get_agent_event().metadata
            else:
                metadata = env.last_event.metadata
            
            scene_info = self.remove_useless_info(metadata)
            subgoals = self.llm_agent.get_subgoals_from_scene(goal_instr, scene_info)

            self.llm_agent.reset_conversation()

            done, success = False, False
            fails = 0
            t = 0
            reward = 0
            action_history = []
            consecutive_fails = 0

            print("Starting stepwise VLM evaluation...")

            while not done and t < args.max_steps:
                frame = self._get_frame(env)

                try:
                    next_action = self.llm_agent.get_next_action(
                        task_desc=goal_instr,
                        subgoals=subgoals,
                        image=frame,
                        action_history=action_history,
                    )
                except Exception as e:
                    self.log(f"Error getting next action: {e}")
                    break

                if next_action.get('action', '').lower() in ['stop', 'end', 'finish', 'done']:
                    print(f"Step {t}: VLM requested STOP")
                    break

                if args.debug:
                    print(f"Step {t}: {next_action}")

                t_success, event, err = self.execute_action(env, next_action, smooth_nav=args.smooth_nav)

                self.log(f"Step {t}: Action: {next_action}, Success: {t_success}, Error: {err}")

                result_object_id = None
                if next_action.get('action') == 'GetObjectInFrame':
                    if event is not None and getattr(event, "metadata", None):
                        result_object_id = event.metadata.get("actionReturn")

                action_record = {
                    'action': next_action.get('action'),
                    'object_id': next_action.get('object_id'),
                    'x': next_action.get('x'),
                    'y': next_action.get('y'),
                    'result_object_id': result_object_id,
                    'success': t_success,
                    'error': err if not t_success else None,
                }
                action_history.append(action_record)

                self.llm_agent.update_action_history(
                    next_action,
                    t_success,
                    err,
                    result_object_id=result_object_id,
                )

                if not t_success:
                    consecutive_fails += 1
                    if consecutive_fails >= args.max_fails:
                        print(f"Too many consecutive failures ({consecutive_fails}). Latest error: {err}")
                        break
                else:
                    consecutive_fails = 0

                t_reward, t_done = env.get_transition_reward()
                reward += t_reward
                t += 1

                if t % 5 == 0:
                    goal_satisfied = env.get_goal_satisfied()
                    if goal_satisfied:
                        print(f"Goal satisfied at step {t}!")
                        success = True
                        done = True

            goal_satisfied = env.get_goal_satisfied()
            if goal_satisfied:
                print("Goal Reached")
                success = True

            pcs = env.get_goal_conditions_met()
            goal_condition_success_rate = pcs[0] / float(pcs[1]) if pcs[1] > 0 else 0

            path_len_weight = len(traj_data['plan']['low_actions'])
            s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / float(t)) if t > 0 else 0
            pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(t)) if t > 0 else 0
            plw_s_spl = s_spl * path_len_weight
            plw_pc_spl = pc_spl * path_len_weight

            lock.acquire()
            try:
                log_entry = {
                    'trial': traj_data['task_id'],
                    'type': traj_data['task_type'],
                    'goal_instr': goal_instr,
                    'completed_goal_conditions': int(pcs[0]),
                    'total_goal_conditions': int(pcs[1]),
                    'goal_condition_success': float(goal_condition_success_rate),
                    'success_spl': float(s_spl),
                    'path_len_weighted_success_spl': float(plw_s_spl),
                    'goal_condition_spl': float(pc_spl),
                    'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
                    'path_len_weight': int(path_len_weight),
                    'reward': float(reward),
                    'executed_actions': t,
                    'stepwise_mode': True,
                    'vlm_mode': True,
                }

                log_entry['trajectory'] = trace.export()
                if success:
                    successes.append(log_entry)
                else:
                    failures.append(log_entry)

                with open(self.trace_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'trajectory': trace.export(),
                        'success': bool(success),
                    }, f, indent=2)
                    print(f"Saved trajectory log to {self.trace_file}")

                results['all'] = self.get_metrics(successes, failures)

                if results.get('all'):
                    print("-------------")
                    print("SR: %d/%d = %.3f" % (results['all']['success']['num_successes'],
                                                results['all']['success']['num_evals'],
                                                results['all']['success']['success_rate']))
                    print("GC: %d/%d = %.3f" % (results['all']['goal_condition_success']['completed_goal_conditions'],
                                                results['all']['goal_condition_success']['total_goal_conditions'],
                                                results['all']['goal_condition_success']['goal_condition_success_rate']))
                    print("-------------")
            finally:
                lock.release()
        finally:
            self._current_trace = previous_trace


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_file', type=str, default=None, help='Path to single trajectory JSON file for testing')
    parser.add_argument('--max_steps', type=int, default=50, help='Maximum steps per episode')
    parser.add_argument('--max_fails', type=int, default=5, help='Maximum consecutive action fails before aborting')
    parser.add_argument('--smooth_nav', action='store_true', help='Use smooth navigation')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
    parser.add_argument('--llm_model', type=str, default='google/gemini-2.5-flash', help='VLM model to use')
    parser.add_argument('--max_tokens', type=int, default=1000, help='Max tokens for VLM response')
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for VLM sampling')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p for VLM sampling')
    parser.add_argument('--frequency_penalty', type=float, default=0.0, help='Frequency penalty for VLM')
    parser.add_argument('--presence_penalty', type=float, default=0.0, help='Presence penalty for VLM')
    parser.add_argument('--reward_config', default='models/config/rewards.json')
    parser.add_argument('--batch', action='store_true', help='Run batch evaluation')
    parser.add_argument('--split', type=str, default='valid_seen', help='Data split to evaluate')
    parser.add_argument('--data_dir', type=str, default='data/json_2.1.0', help='Data directory')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs per trajectory')
    parser.add_argument('--ridx', type=int, default=0, nargs='?', help='Repeat index for single trajectory test')

    args = parser.parse_args()

    evaluator = EvalVLMStepwise(args)
    if args.batch:
        evaluator.test_batch(args.data_dir, args.split, args.num_runs)
    else:
        # Note: goto=True is implied by using EvalLLMAstar features, but parameter passed to evaluate()
        evaluator.test_single_trajectory(args.traj_file, goto=True, r_idx=args.ridx)
