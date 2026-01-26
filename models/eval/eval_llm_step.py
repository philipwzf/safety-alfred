import json
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.eval.eval_llm import EpisodeTrace
from models.eval.eval_llm_astar import EvalLLMAstar
from models.model.llm_step import LLM_StepAgent


class EvalLLMStepwise(EvalLLMAstar):
    """
    Stepwise LLM evaluation (text-only, no perception tools).
    
    Inherits from EvalLLMAstar to support GotoLocation navigation via A* pathfinding.
    Uses LLM_StepAgent for stepwise action generation based on scene info.
    """
    
    def __init__(self, args, manager=None):
        # Call parent constructor for basic setup (including Graph for A* navigation)
        super().__init__(args, manager)
        
        # Replace the LLM agent with stepwise version
        self.llm_agent = LLM_StepAgent(args)
        self.llm_agent.set_log_method(self.log)
        print(f"Stepwise LLM logging to: {self.log_file}")

    def get_scene_info(self, env, traj_data):
        """
        Convert environment metadata to the flattened format expected by LLM_StepAgent.create_prompt().
        
        This method bridges the gap between ai2thor's nested metadata structure and 
        the flat key format used in the stepwise prompt template.
        """
        metadata = getattr(env.last_event, "metadata", None) if env is not None else None
        if metadata is None:
            return {}
        
        # Use inherited remove_useless_info to clean up metadata
        trimmed = self.remove_useless_info(metadata)
        
        # Extract agent info from nested structure
        agent_meta = trimmed.get("agent", {}) or {}
        
        # Get held object from inventory
        inventory = trimmed.get("inventoryObjects") or []
        held_object = inventory[0] if inventory else None
        
        # Get scene number from traj_data
        scene_num = None
        if isinstance(traj_data, dict):
            scene_info = traj_data.get("scene") or {}
            scene_num = scene_info.get("scene_num", traj_data.get("scene_num"))
        
        # Return flattened structure matching LLM_StepAgent.create_prompt() expectations
        return {
            "agent_position": agent_meta.get("position", {}),
            "agent_rotation": agent_meta.get("rotation", {}),
            "scene_num": scene_num,
            "agent_held_object": held_object,
            "objects": trimmed.get("objects", []),
        }

    def evaluate(self, env, r_idx, traj_data, args, lock, successes, failures, results, goto=False, inject_danger=False):
        """
        Override the main evaluation method for stepwise LLM execution.
        
        Args:
            env: ThorEnv instance
            r_idx: Repeat index for trajectory annotation
            traj_data: Trajectory data dict
            args: Command line arguments
            lock: Thread lock for result aggregation
            successes: List to append successful evaluations
            failures: List to append failed evaluations
            results: Dict to store aggregated metrics
            goto: Whether to use GotoLocation (always True for stepwise, inherited from EvalLLMAstar)
            inject_danger: Whether to inject danger objects into scene
        """
        trace = EpisodeTrace()
        previous_trace = self._current_trace
        self._current_trace = trace
        
        try:
            # Setup scene (inherited from parent)
            reward_type = 'sparse'
            self.setup_scene(env, traj_data, r_idx, args, reward_type=reward_type, inject_danger=inject_danger)

            # Goal instruction - handle both formats
            goal_instr = traj_data.get('task_desc')
            if not goal_instr:
                anns = traj_data.get('turk_annotations', {}).get('anns', [])
                if anns and r_idx < len(anns):
                    goal_instr = anns[r_idx].get('task_desc')
            goal_instr = goal_instr or ''

            # Log task info
            self.log(f"Task: {goal_instr}")
            self.log(f"Scene: {traj_data['scene']['scene_num']}")

            # Reset conversation for new episode
            self.llm_agent.reset_conversation()

            # Initialize tracking variables
            done, success = False, False
            t = 0
            reward = 0
            action_history = []
            consecutive_fails = 0

            print("Starting stepwise LLM evaluation...")
            
            # Main stepwise loop
            while not done and t < args.max_steps:
                # Get current scene info in flattened format
                scene_info = self.get_scene_info(env, traj_data)
                
                # Get next action from stepwise LLM agent
                try:
                    next_action = self.llm_agent.get_next_action(
                        task_desc=goal_instr,
                        scene_info=scene_info,
                        action_history=action_history
                    )
                except Exception as e:
                    self.log(f"Error getting next action: {e}")
                    break
                
                # Check if stop action
                if next_action.get('action', '').lower() in ['stop', 'end', 'finish', 'done']:
                    print(f"Step {t}: LLM requested STOP")
                    break

                # Print action for debugging
                if args.debug:
                    print(f"Step {t}: {next_action}")

                # Execute action (inherited from EvalLLMAstar, supports GotoLocation)
                t_success, event, err = self.execute_action(env, next_action, smooth_nav=args.smooth_nav)
                
                # Log action execution
                self.log(f"Step {t}: Action: {next_action}, Success: {t_success}, Error: {err}")
                
                # Update action history for next LLM call
                action_record = {
                    'action': next_action.get('action'),
                    'object_id': next_action.get('object_id'),
                    'success': t_success,
                    'error': err if not t_success else None
                }
                action_history.append(action_record)
                
                # Update LLM agent's internal history
                self.llm_agent.update_action_history(next_action, t_success, err)
                
                # Handle failures
                if not t_success:
                    consecutive_fails += 1
                    if consecutive_fails >= args.max_fails:
                        print(f"Too many consecutive failures ({consecutive_fails}). Latest error: {err}")
                        break
                else:
                    consecutive_fails = 0  # Reset on success

                # Get reward and check if done
                t_reward, t_done = env.get_transition_reward()
                reward += t_reward
                t += 1

                # Check goal satisfaction periodically
                if t % 5 == 0:
                    goal_satisfied = env.get_goal_satisfied()
                    if goal_satisfied:
                        print(f"Goal satisfied at step {t}!")
                        success = True
                        done = True

            # Final goal check
            goal_satisfied = env.get_goal_satisfied()
            if goal_satisfied:
                print("Goal Reached")
                success = True

            # Calculate metrics
            pcs = env.get_goal_conditions_met()
            goal_condition_success_rate = pcs[0] / float(pcs[1]) if pcs[1] > 0 else 0

            # SPL calculation
            path_len_weight = len(traj_data['plan']['low_actions'])
            s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / float(t)) if t > 0 else 0
            pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(t)) if t > 0 else 0

            # Path length weighted SPL
            plw_s_spl = s_spl * path_len_weight
            plw_pc_spl = pc_spl * path_len_weight

            # Log results with thread safety
            lock.acquire()
            try:
                log_entry = {
                    'trial': traj_data['task_id'],
                    'type': traj_data['task_type'],
                    'repeat_idx': int(r_idx),
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
                    'llm_mode': True,
                }
                
                log_entry['trajectory'] = trace.export()
                if success:
                    successes.append(log_entry)
                else:
                    failures.append(log_entry)

                # Save trajectory trace
                with open(self.trace_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'trajectory': trace.export(),
                        'success': bool(success),
                    }, f, indent=2)
                    print(f"Saved trajectory log to {self.trace_file}")

                # Overall results
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
    parser.add_argument('--max_steps', type=int, default=25, help='Maximum steps per episode')
    parser.add_argument('--max_fails', type=int, default=5, help='Maximum consecutive action fails before aborting')
    parser.add_argument('--smooth_nav', action='store_true', help='Use smooth navigation')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
    parser.add_argument('--llm_model', type=str, default='deepseek/deepseek-chat', help='LLM model to use')
    parser.add_argument('--max_tokens', type=int, default=10000, help='Max tokens for LLM response')
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for LLM sampling')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p for LLM sampling')
    parser.add_argument('--frequency_penalty', type=float, default=0.0, help='Frequency penalty for LLM')
    parser.add_argument('--presence_penalty', type=float, default=0.0, help='Presence penalty for LLM')
    parser.add_argument('--reward_config', default='models/config/rewards.json')
    parser.add_argument('--batch', action='store_true', help='Run batch evaluation')
    parser.add_argument('--split', type=str, default='valid_seen', help='Data split to evaluate')
    parser.add_argument('--data_dir', type=str, default='data/json_2.1.0', help='Data directory')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs per trajectory')
    parser.add_argument('--ridx', type=int, default=0, nargs='?', help='Repeat index for single trajectory test')
    parser.add_argument('--setup_debug', action='store_true', help='Log only setup issues for debugging scene restoration')

    args = parser.parse_args()
    
    evaluator = EvalLLMStepwise(args)
    if args.batch:
        evaluator.test_batch(args.data_dir, args.split, args.num_runs)
    else:
        evaluator.test_single_trajectory(args.traj_file, goto=True, r_idx=args.ridx, inject_danger=True)
