import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from models.eval.eval_llm import EpisodeTrace
from models.eval.eval_llm_step import EvalLLMStepwise
from models.model.safety_critic import SafetyCritic


class EvalLLMReact(EvalLLMStepwise):
    """Stepwise LLM rollout with one safety-feedback revision per step."""

    def __init__(self, args, manager=None):
        super().__init__(args, manager)
        self.safety_critic = SafetyCritic()
        self.react_feedback_attempts = 1
        print(f"ReAct safety feedback logging to: {self.log_file}")

    def _current_metadata(self, env):
        event = getattr(env, "last_event", None)
        metadata = getattr(event, "metadata", None)
        return metadata or {}

    def _get_action_with_feedback(self, goal_instr, scene_info, action_history, metadata):
        candidate_action = self.llm_agent.get_next_action(
            task_desc=goal_instr,
            scene_info=scene_info,
            action_history=action_history,
        )
        candidate_feedback = self.safety_critic.check(candidate_action, metadata)
        record = {
            "candidate_action": candidate_action,
            "candidate_safe": candidate_feedback.safe,
            "feedback": candidate_feedback.to_prompt(),
            "revised_action": None,
            "revised_safe": None,
        }

        if candidate_feedback.safe:
            return candidate_action, None, record

        feedback_prompt = candidate_feedback.to_prompt()
        self.log("SAFETY FEEDBACK:")
        self.log(feedback_prompt)

        revised_action = self.llm_agent.get_next_action(
            task_desc=goal_instr,
            scene_info=scene_info,
            action_history=action_history,
            safety_feedback=feedback_prompt,
        )
        revised_feedback = self.safety_critic.check(revised_action, metadata)
        record["revised_action"] = revised_action
        record["revised_safe"] = revised_feedback.safe
        if not revised_feedback.safe:
            record["revised_feedback"] = revised_feedback.to_prompt()
            self.log("REVISED ACTION STILL FLAGGED UNSAFE:")
            self.log(revised_feedback.to_prompt())

        return revised_action, feedback_prompt, record

    def evaluate(self, env, r_idx, traj_data, args, lock, successes, failures, results, goto=False, inject_danger=False):
        trace = EpisodeTrace()
        previous_trace = self._current_trace
        self._current_trace = trace
        react_records = []
        try:
            self.setup_scene(env, traj_data, r_idx, args, reward_type='dense', inject_danger=inject_danger)

            anns = traj_data.get('turk_annotations', {}).get('anns', [])
            goal_instr = anns[r_idx].get('task_desc') if len(anns) > r_idx else ''
            goal_instr = traj_data.get('task_desc') or goal_instr

            self.log(f"Task: {goal_instr}")
            self.log(f"Scene: {traj_data['scene']['scene_num']}")
            self.llm_agent.reset_conversation()

            done, success = False, False
            t = 0
            reward = 0
            action_history = []
            consecutive_fails = 0

            print("Starting ReAct stepwise evaluation...")

            while not done and t < args.max_steps:
                scene_info = self.get_scene_info(env, traj_data)
                metadata = self._current_metadata(env)

                try:
                    next_action, safety_feedback, react_record = self._get_action_with_feedback(
                        goal_instr, scene_info, action_history, metadata
                    )
                except Exception as exc:
                    self.log(f"Error getting next ReAct action: {exc}")
                    raise

                react_record["step"] = t
                react_records.append(react_record)

                requested_stop = self._is_stop_action(next_action)
                if requested_stop:
                    print(f"Step {t}: LLM requested STOP")
                    next_action = {'action': 'Done'}

                if args.debug:
                    print(f"Step {t}: {next_action}")

                react_record["executed_action"] = next_action
                t_success, event, err = self.execute_action(env, next_action, smooth_nav=args.smooth_nav)
                react_record["execution_success"] = bool(t_success)
                react_record["execution_error"] = err if not t_success else None
                self.log(f"Step {t}: Action: {next_action}, Success: {t_success}, Error: {err}")

                action_record = {
                    'action': next_action.get('action'),
                    'object_id': next_action.get('object_id'),
                    'success': t_success,
                    'error': err if not t_success else None,
                }
                if safety_feedback:
                    action_record['safety_feedback'] = safety_feedback
                action_history.append(action_record)
                self.llm_agent.update_action_history(next_action, t_success, err, safety_feedback=safety_feedback)

                if not t_success:
                    consecutive_fails += 1
                    if consecutive_fails >= args.max_fails:
                        print(f"Too many consecutive failures ({consecutive_fails}). Latest error: {err}")
                        break
                else:
                    consecutive_fails = 0

                try:
                    t_reward, _ = env.get_transition_reward()
                    reward += t_reward
                except Exception:
                    pass

                t += 1
                if requested_stop:
                    done = True
                    break

                if t % 5 == 0 and env.get_goal_satisfied():
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
                    'react_mode': True,
                    'react_feedback_count': sum(1 for record in react_records if not record["candidate_safe"]),
                    'trajectory': trace.export(),
                }

                if success:
                    successes.append(log_entry)
                else:
                    failures.append(log_entry)

                with open(self.trace_file, 'w', encoding='utf-8') as handle:
                    react_summary = {
                        "enabled": True,
                        "mode": "soft_pre_action_feedback",
                        "feedback_attempts_per_step": self.react_feedback_attempts,
                        "feedback_count": sum(
                            1 for record in react_records if not record["candidate_safe"]
                        ),
                        "unsafe_after_retry_count": sum(
                            1 for record in react_records if record.get("revised_safe") is False
                        ),
                        "records": EpisodeTrace._sanitize(react_records),
                    }
                    json.dump({
                        'trajectory': trace.export(),
                        'success': bool(success),
                        'react': react_summary,
                    }, handle, indent=2)
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

    @staticmethod
    def _is_stop_action(action):
        return str((action or {}).get('action', '')).lower() in ['stop', 'end', 'finish', 'done']


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_file', type=str, default=None, help='Path to single trajectory JSON file for testing')
    parser.add_argument('--max_steps', type=int, default=50, help='Maximum steps per episode')
    parser.add_argument('--max_fails', type=int, default=5, help='Maximum consecutive action fails before aborting')
    parser.add_argument('--smooth_nav', action='store_true', help='Use smooth navigation')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
    parser.add_argument('--model', type=str, default=None, help='LLM model to use')
    parser.add_argument('--llm_model', type=str, default='deepseek/deepseek-chat',
                        help='Backward-compatible LLM model argument')
    parser.add_argument('--log-model-name', type=str, default=None,
                        help='Optional log namespace; does not change the API model')
    parser.add_argument('--max_tokens', type=int, default=1000, help='Max tokens for LLM response')
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for LLM sampling')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p for LLM sampling')
    parser.add_argument('--frequency_penalty', type=float, default=0.0, help='Frequency penalty for LLM')
    parser.add_argument('--presence_penalty', type=float, default=0.0, help='Presence penalty for LLM')
    parser.add_argument('--reward_config', default='models/config/rewards.json')
    parser.add_argument('--ridx', type=int, default=0, nargs='?', help='Repeat index for single trajectory test')
    parser.add_argument('--inject-danger', action='store_true',
                        help='Explicitly inject additional liquid hazards into the restored scene')
    parser.add_argument('--setup_debug', action='store_true',
                        help='Log only setup issues for debugging scene restoration')

    args = parser.parse_args()

    evaluator = EvalLLMReact(args)
    evaluator.test_single_trajectory(
        args.traj_file,
        goto=True,
        r_idx=args.ridx,
        inject_danger=args.inject_danger,
    )
