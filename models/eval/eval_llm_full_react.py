import json
import os
import sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.eval.eval_llm import EpisodeTrace
from models.eval.eval_llm_astar import EvalLLMAstar
from models.model.trace_level_critic import (
    TraceLevelLLMJudge,
    TraceLevelSentinelCritic,
    intermediate_trace_dir_for,
)


class EvalLLMFullEpisodeReact(EvalLLMAstar):
    """Full-episode trace-level ReAct loop.

    Each loop starts from the original task state, executes a complete plan,
    evaluates the raw trace, and replans with trace-level feedback until the
    rollout is both safe and successful or the loop limit is reached.
    """

    def __init__(self, args, manager=None):
        super().__init__(args, manager)
        self.critic_type = getattr(args, "critic_type", None) or "deterministic"
        self.max_loop_limit = max(1, int(getattr(args, "max_loop_limit", 3)))
        self.constraints_json = getattr(args, "constraints_json", "safety_rules_object.json")
        if self.critic_type == "deterministic":
            self.trace_critic = TraceLevelSentinelCritic(self.constraints_json)
        elif self.critic_type == "llm":
            self.trace_critic = TraceLevelLLMJudge(args, self.constraints_json)
            self.trace_critic.set_log_method(self.log)
        else:
            raise ValueError(f"Unsupported critic_type: {self.critic_type}")
        print(
            f"Full-episode ReAct logging to: {self.log_file} "
            f"(critic_type={self.critic_type}, max_loop_limit={self.max_loop_limit})"
        )

    def _intermediate_dir(self) -> Path:
        return intermediate_trace_dir_for(self.trace_file)

    def _save_intermediate_trace(
        self,
        loop_index,
        trajectory,
        success,
        goal_conditions,
        llm_plan,
        critic_result,
        safe_success,
    ) -> str:
        out_dir = self._intermediate_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"round_{loop_index:02d}.json"
        completed, total = goal_conditions
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "trajectory": trajectory,
                    "success": bool(success),
                    "full_episode_round": {
                        "loop_index": loop_index,
                        "critic_type": self.critic_type,
                        "safe": bool(critic_result.safe),
                        "safe_success": bool(safe_success),
                        "completed_goal_conditions": int(completed),
                        "total_goal_conditions": int(total),
                        "plan": EpisodeTrace._sanitize(llm_plan),
                        "violations": EpisodeTrace._sanitize(critic_result.violations),
                        "errors": EpisodeTrace._sanitize(critic_result.errors),
                        "raw_response": critic_result.raw_response,
                    },
                },
                handle,
                indent=2,
            )
        return str(out_path)

    def _run_full_episode_round(self, env, r_idx, traj_data, args, goto, safety_feedback, inject_danger):
        trace = EpisodeTrace()
        previous_trace = self._current_trace
        self._current_trace = trace
        try:
            self.setup_scene(env, traj_data, r_idx, args, reward_type='sparse', inject_danger=inject_danger)

            anns = traj_data.get('turk_annotations', {}).get('anns', [])
            goal_instr = anns[r_idx].get('task_desc') if len(anns) > r_idx else ''
            goal_instr = traj_data.get('task_desc') or goal_instr
            self.log(f"Task: {goal_instr}")
            self.log(f"Scene: {traj_data['scene']['scene_num']}")

            metadata = env.last_event.metadata
            scene_info = self.remove_useless_info(metadata)
            subgoals = self.llm_agent.get_subgoals_from_scene(
                goal_instr,
                scene_info,
                safety_feedback=safety_feedback,
            )
            llm_plan = self.llm_agent.generate_plan(
                subgoals,
                scene_info,
                goto=goto,
                safety_feedback=safety_feedback,
            )

            fails = 0
            t = 0
            action_idx = 0
            print(f"Generated plan with {len(llm_plan)} actions")

            while action_idx < len(llm_plan):
                if t >= args.max_steps:
                    print("Max steps reached")
                    break

                action_data = llm_plan[action_idx]
                action = action_data.get('action')
                if not action:
                    print("Invalid action in plan")
                    break

                if action.lower() in ['stop', 'end', 'finish', 'done']:
                    print("\tLLM predicted STOP")
                    self.execute_action(env, {'action': 'Done'}, smooth_nav=args.smooth_nav)
                    break

                if args.debug:
                    print(f"Step {t}: {action_data}")

                t_success, _event, err = self.execute_action(
                    env,
                    action_data,
                    smooth_nav=args.smooth_nav,
                )
                if not t_success:
                    fails += 1
                    if fails >= args.max_fails:
                        print(
                            "Interact API failed %d times" % fails
                            + "; latest error '%s'" % err
                        )
                        break

                t += 1
                action_idx += 1

            success = bool(env.get_goal_satisfied())
            if success:
                print("Goal Reached")

            pcs = env.get_goal_conditions_met()
            return {
                "goal_instr": goal_instr,
                "llm_plan": llm_plan,
                "trajectory": trace.export(),
                "success": success,
                "goal_conditions": (int(pcs[0]), int(pcs[1])),
                "steps_failed": int(fails),
                "executed_actions": int(t),
            }
        finally:
            self._current_trace = previous_trace

    def evaluate(self, env, r_idx, traj_data, args, lock, successes, failures, results, goto=False, inject_danger=False):
        loop_records = []
        feedback = None
        final_round = None

        for loop_index in range(1, self.max_loop_limit + 1):
            print(f"Starting full-episode ReAct loop {loop_index}/{self.max_loop_limit}")
            round_result = self._run_full_episode_round(
                env,
                r_idx,
                traj_data,
                args,
                goto=goto,
                safety_feedback=feedback,
                inject_danger=inject_danger,
            )
            critic_result = self.trace_critic.evaluate(
                round_result["trajectory"],
                task_desc=round_result["goal_instr"],
                env_success=round_result["success"],
                goal_conditions=round_result["goal_conditions"],
                loop_index=loop_index,
            )
            safe_success = bool(critic_result.safe and round_result["success"])
            intermediate_path = self._save_intermediate_trace(
                loop_index,
                round_result["trajectory"],
                round_result["success"],
                round_result["goal_conditions"],
                round_result["llm_plan"],
                critic_result,
                safe_success,
            )

            loop_record = {
                "loop_index": loop_index,
                "critic_type": self.critic_type,
                "safe": bool(critic_result.safe),
                "success": bool(round_result["success"]),
                "safe_success": safe_success,
                "completed_goal_conditions": round_result["goal_conditions"][0],
                "total_goal_conditions": round_result["goal_conditions"][1],
                "llm_plan_length": len(round_result["llm_plan"]),
                "steps_failed": round_result["steps_failed"],
                "executed_actions": round_result["executed_actions"],
                "intermediate_trace": intermediate_path,
                "violations": EpisodeTrace._sanitize(critic_result.violations),
                "errors": EpisodeTrace._sanitize(critic_result.errors),
                "feedback": critic_result.feedback,
            }
            if critic_result.raw_response:
                loop_record["raw_response"] = critic_result.raw_response
            loop_records.append(loop_record)
            final_round = (round_result, critic_result, safe_success)

            if safe_success:
                print(f"Safe & Success reached at loop {loop_index}.")
                break

            feedback = critic_result.feedback
            self.log("FULL-EPISODE REACT FEEDBACK:")
            self.log(feedback)

        if final_round is None:
            raise RuntimeError("Full-episode ReAct produced no rollout rounds")

        round_result, critic_result, safe_success = final_round
        completed, total = round_result["goal_conditions"]
        goal_condition_success_rate = completed / float(total) if total > 0 else 0.0
        path_len_weight = len(traj_data.get('plan', {}).get('low_actions', []) or [])

        lock.acquire()
        try:
            log_entry = {
                'trial': traj_data['task_id'],
                'repeat_idx': int(r_idx),
                'goal_instr': round_result["goal_instr"],
                'completed_goal_conditions': int(completed),
                'total_goal_conditions': int(total),
                'goal_condition_success': float(goal_condition_success_rate),
                'llm_plan_length': len(round_result["llm_plan"]),
                'steps_failed': int(round_result["steps_failed"]),
                'path_len_weight': int(path_len_weight),
                'trajectory': round_result["trajectory"],
                'react_mode': True,
                'full_episode_react': True,
                'safe_success': safe_success,
            }

            with open(self.trace_file, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'trajectory': round_result["trajectory"],
                        'success': bool(round_result["success"]),
                        'full_episode_react': {
                            "enabled": True,
                            "critic_type": self.critic_type,
                            "constraints_json": self.constraints_json,
                            "max_loop_limit": self.max_loop_limit,
                            "loops_run": len(loop_records),
                            "final_safe": bool(critic_result.safe),
                            "final_success": bool(round_result["success"]),
                            "final_safe_success": bool(safe_success),
                            "records": EpisodeTrace._sanitize(loop_records),
                        },
                    },
                    f,
                    indent=2,
                )
                print(f"Saved trajectory log to {self.trace_file}")

            if round_result["success"]:
                successes.append(log_entry)
            else:
                failures.append(log_entry)

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_file', type=str, default=None)
    parser.add_argument('--max_steps', type=int, default=50)
    parser.add_argument('--max_fails', type=int, default=5)
    parser.add_argument('--max-loop-limit', dest='max_loop_limit', type=int, default=3)
    parser.add_argument('--smooth_nav', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--reward_config', default='models/config/rewards.json')
    parser.add_argument('--model', type=str, default=None, help='LLM model to use')
    parser.add_argument('--llm_model', type=str, default='deepseek/deepseek-chat',
                        help='Backward-compatible LLM model argument')
    parser.add_argument('--log-model-name', type=str, default=None,
                        help='Optional log namespace; does not change the API model')
    parser.add_argument('--critic-type', dest='critic_type',
                        choices=['deterministic', 'llm'], default='deterministic')
    parser.add_argument('--constraints-json', dest='constraints_json',
                        default='safety_rules_object.json')
    parser.add_argument('--max_tokens', type=int, default=10000, help='Max tokens for LLM response')
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for LLM sampling')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p for LLM sampling')
    parser.add_argument('--frequency_penalty', type=float, default=0.0, help='Frequency penalty for LLM')
    parser.add_argument('--presence_penalty', type=float, default=0.0, help='Presence penalty for LLM')
    parser.add_argument('--ridx', type=int, default=0, nargs='?',
                        help='Repeat index for single trajectory test')
    parser.add_argument('--inject-danger', action='store_true', dest='inject_danger',
                        help='Explicitly inject additional liquid hazards into the restored scene')
    parser.add_argument('--setup_debug', action='store_true',
                        help='Log only setup issues for debugging scene restoration')

    args = parser.parse_args()

    # Mirror existing safety-alfred convention: fall back from --llm_model when
    # --model is not provided, so TraceLevelLLMJudge.args.model is never None.
    if not getattr(args, 'model', None):
        args.model = getattr(args, 'llm_model', None)

    evaluator = EvalLLMFullEpisodeReact(args)
    evaluator.test_single_trajectory(
        args.traj_file,
        goto=True,
        r_idx=args.ridx,
        inject_danger=args.inject_danger,
    )
