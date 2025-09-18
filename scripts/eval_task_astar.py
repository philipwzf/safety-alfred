"""Run EvalLLMAStar over all pick_and_place_simple train tasks with detailed logging."""

import argparse
import glob
import json
import os
import queue
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

# Ensure project root is on path when executed from scripts/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.thor_env import ThorEnv
from models.eval.eval_llm_astar import EvalLLMAstar


@dataclass
class EpisodeRecord:
    """Container for a single evaluation episode."""

    data: Dict

    @property
    def success(self) -> bool:
        return bool(self.data.get("success", False))

    @property
    def goal_conditions(self) -> int:
        return int(self.data.get("total_goal_conditions", 0))

    @property
    def completed_goal_conditions(self) -> int:
        return int(self.data.get("completed_goal_conditions", 0))

    def to_dict(self) -> Dict:
        return self.data


class EpisodeLogger:
    """Streams per-episode records to disk while keeping an in-memory copy."""

    def __init__(self, output_dir: str, timestamp: str) -> None:
        self.records: List[Dict] = []
        self.timestamp = timestamp
        self.jsonl_path = os.path.join(
            output_dir, f"episode_records_{timestamp}.jsonl"
        )
        self.json_path = os.path.join(
            output_dir, f"episode_records_{timestamp}.json"
        )
        self._stream = open(self.jsonl_path, "w", encoding="utf-8")

    def log(self, record: Dict) -> None:
        json.dump(record, self._stream)
        self._stream.write("\n")
        self._stream.flush()
        self.records.append(record)

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

class _SimpleLock:
    """Minimal lock compatible with EvalLLM.evaluate."""

    def acquire(self) -> None:  # pragma: no cover - trivial
        return

    def release(self) -> None:  # pragma: no cover - trivial
        return


def _chunk_sequence(seq: Sequence[str], num_chunks: int) -> List[List[str]]:
    if num_chunks <= 1:
        return [list(seq)]
    n = len(seq)
    if n == 0:
        return [[] for _ in range(num_chunks)]
    base = n // num_chunks
    remainder = n % num_chunks
    chunks: List[List[str]] = []
    start = 0
    for i in range(num_chunks):
        extra = 1 if i < remainder else 0
        end = start + base + extra
        chunks.append(list(seq[start:end]))
        start = end
    return chunks


def _add_metadata(entry: Dict, traj_data: Dict, traj_file: str, annotation_idx: int,
                  worker_id: int, gpu_id: Optional[int], duration: float) -> None:
    entry.update({
        "traj_file": traj_file,
        "annotation_idx": annotation_idx,
        "scene_id": traj_data.get("scene", {}).get("scene_num"),
        "object_target": traj_data.get("pddl_params", {}).get("object_target"),
        "parent_target": traj_data.get("pddl_params", {}).get("parent_target"),
        "gpu_id": gpu_id,
        "worker_id": worker_id,
        "duration_sec": duration,
    })
    # Keep backward compatibility with EvalLLM metrics
    entry.setdefault("type", traj_data.get("task_type"))
    entry.setdefault("trial", traj_data.get("task_id"))


def _evaluate_episode(evaluator: EvalLLMAstar, env: ThorEnv, traj_data: Dict,
                      annotation_idx: int, args, traj_file: str,
                      worker_id: int, gpu_id: Optional[int]) -> EpisodeRecord:
    start_time = time.time()
    successes: List[Dict] = []
    failures: List[Dict] = []
    results: Dict = {}

    evaluator.evaluate(env, annotation_idx, traj_data, args, _SimpleLock(),
                       successes, failures, results, goto=True)

    duration = time.time() - start_time
    if successes:
        entry = successes[-1]
        entry["success"] = True
    elif failures:
        entry = failures[-1]
        entry["success"] = False
    else:
        # Should not happen, but guard against it by creating a stub failure entry
        entry = {
            "success": False,
            "completed_goal_conditions": 0,
            "total_goal_conditions": 0,
            "success_spl": 0.0,
            "path_len_weighted_success_spl": 0.0,
            "goal_condition_spl": 0.0,
            "path_len_weighted_goal_condition_spl": 0.0,
            "path_len_weight": 0,
            "reward": 0.0,
            "llm_plan_length": 0,
            "executed_actions": 0,
        }

    _add_metadata(entry, traj_data, traj_file, annotation_idx, worker_id, gpu_id, duration)
    return EpisodeRecord(entry)


def _process_traj_files(worker_id: int, traj_files: Sequence[str], args, annotation_limit: int,
                        gpu_id: Optional[int], record_callback: Callable[[Dict], None]) -> None:
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    evaluator = EvalLLMAstar(args)
    env = ThorEnv()
    try:
        for traj_file in traj_files:
            with open(traj_file, "r", encoding="utf-8") as f:
                traj_data = json.load(f)
            annotations = traj_data.get("turk_annotations", {}).get("anns", [])
            num_annotations = len(annotations)
            if num_annotations == 0:
                continue
            limit = annotation_limit if annotation_limit > 0 else num_annotations
            for annotation_idx in range(min(limit, num_annotations)):
                try:
                    record = _evaluate_episode(
                        evaluator, env, traj_data, annotation_idx, args,
                        traj_file, worker_id, gpu_id
                    )
                except Exception as exc:  # pragma: no cover - runtime safeguard
                    record = EpisodeRecord({
                        "success": False,
                        "error": str(exc),
                        "completed_goal_conditions": 0,
                        "total_goal_conditions": 0,
                        "success_spl": 0.0,
                        "path_len_weighted_success_spl": 0.0,
                        "goal_condition_spl": 0.0,
                        "path_len_weighted_goal_condition_spl": 0.0,
                        "path_len_weight": 0,
                        "reward": 0.0,
                        "llm_plan_length": 0,
                        "executed_actions": 0,
                    })
                    _add_metadata(record.data, traj_data, traj_file, annotation_idx,
                                  worker_id, gpu_id, 0.0)
                record_callback(record.to_dict())
    finally:
        env.stop()


def _worker_main(worker_id: int, traj_files: Sequence[str], args, annotation_limit: int,
                 gpu_id: Optional[int], result_queue) -> None:
    def enqueue(record: Dict) -> None:
        result_queue.put(record)

    try:
        _process_traj_files(worker_id, traj_files, args, annotation_limit, gpu_id, enqueue)
    finally:
        result_queue.put(("done", worker_id))


def _summarize(records: List[EpisodeRecord]) -> Dict:
    if not records:
        return {
            "num_episodes": 0,
            "num_successes": 0,
            "num_failures": 0,
            "success_rate": 0.0,
            "goal_condition_success_rate": 0.0,
            "avg_success_spl": 0.0,
            "avg_goal_condition_spl": 0.0,
        }

    successes = [rec.to_dict() for rec in records if rec.success]
    failures = [rec.to_dict() for rec in records if not rec.success]
    metrics = EvalLLMAstar.get_metrics(successes, failures)

    total_episodes = len(records)
    total_goal = sum(rec.goal_conditions for rec in records)
    total_goal_completed = sum(rec.completed_goal_conditions for rec in records)

    avg = lambda key: sum(float(rec.to_dict().get(key, 0.0)) for rec in records) / total_episodes
    total_duration = sum(float(rec.to_dict().get("duration_sec", 0.0)) for rec in records)

    summary = {
        "num_episodes": total_episodes,
        "num_successes": len(successes),
        "num_failures": len(failures),
        "success_rate": metrics.get("success", {}).get("success_rate", 0.0) if metrics else 0.0,
        "goal_condition_success_rate": metrics.get("goal_condition_success", {}).get(
            "goal_condition_success_rate", 0.0
        ) if metrics else 0.0,
        "path_length_weighted_success_rate": metrics.get(
            "path_length_weighted_success_rate", 0.0
        ) if metrics else 0.0,
        "path_length_weighted_goal_condition_success_rate": metrics.get(
            "path_length_weighted_goal_condition_success_rate", 0.0
        ) if metrics else 0.0,
        "avg_success_spl": avg("success_spl"),
        "avg_goal_condition_spl": avg("goal_condition_spl"),
        "avg_reward": avg("reward"),
        "avg_plan_length": avg("llm_plan_length"),
        "avg_executed_actions": avg("executed_actions"),
        "avg_path_length_weight": avg("path_len_weight"),
        "avg_duration_sec": total_duration / total_episodes,
        "total_duration_sec": total_duration,
        "total_goal_conditions": total_goal,
        "completed_goal_conditions": total_goal_completed,
    }
    return summary


def _breakdown(records: List[EpisodeRecord], key: str) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {
        "successes": 0,
        "failures": 0,
        "total": 0,
        "completed_goal_conditions": 0,
        "total_goal_conditions": 0,
        "avg_reward": 0.0,
        "avg_duration_sec": 0.0,
    })

    for rec in records:
        rec_dict = rec.to_dict()
        value = rec_dict.get(key)
        if value is None or value == "":
            continue
        entry = stats[str(value)]
        entry["total"] += 1
        entry["avg_reward"] += float(rec_dict.get("reward", 0.0))
        entry["avg_duration_sec"] += float(rec_dict.get("duration_sec", 0.0))
        entry["completed_goal_conditions"] += rec.completed_goal_conditions
        entry["total_goal_conditions"] += rec.goal_conditions
        if rec.success:
            entry["successes"] += 1
        else:
            entry["failures"] += 1

    for data in stats.values():
        total = data["total"]
        if total > 0:
            data["success_rate"] = data["successes"] / total
            data["goal_condition_success_rate"] = (
                data["completed_goal_conditions"] / data["total_goal_conditions"]
                if data["total_goal_conditions"] > 0 else 0.0
            )
            data["avg_reward"] /= total
            data["avg_duration_sec"] /= total
        else:
            data["success_rate"] = 0.0
            data["goal_condition_success_rate"] = 0.0
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate train tasks (by task name) with EvalLLMAstar"
    )
    ## Here are the task names to pick from:
    # pick_and_place_simple
    # pick_two_objs_and_place
    # pick_and_place_with_movable_recep
    # pick_heat_then_place_in_recep
    # pick_cool_then_place_in_recep
    # pick_clean_then_place_in_recep
    # look_at_obj_in_light
    parser.add_argument("--task", default="pick_and_place_simple", help="Task name to evaluate")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of parallel evaluation workers")
    parser.add_argument("--gpu-ids", type=int, nargs="*", default=None,
                        help="Optional list of GPU ids to assign to workers")
    parser.add_argument("--annotations-per-task", type=int, default=1,
                        help="How many instructions per trajectory to evaluate (0 = all)")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--max-fails", type=int, default=5,
                        help="Maximum consecutive action failures before aborting")
    parser.add_argument("--smooth-nav", action="store_true", help="Use smooth navigation primitives")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug output")
    parser.add_argument("--llm-model", type=str, default="deepseek/deepseek-chat",
                        help="LLM model identifier")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Max LLM tokens")
    parser.add_argument("--temperature", type=float, default=0.6, help="LLM sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="LLM nucleus sampling")
    parser.add_argument("--frequency-penalty", type=float, default=0.0,
                        help="LLM frequency penalty")
    parser.add_argument("--presence-penalty", type=float, default=0.0,
                        help="LLM presence penalty")
    parser.add_argument("--reward-config", default="models/config/rewards.json",
                        help="Reward configuration file for ThorEnv")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on number of trajectories to evaluate")
    parser.add_argument("--resume", type=str, default=None,
                        help="Optional JSON file with previously evaluated trajectory ids to skip")
    parser.add_argument("--no-multiprocessing", action="store_true",
                        help="Run in the main process regardless of num-workers")
    return parser.parse_args()


def load_resume_set(resume_file: Optional[str]) -> Optional[set]:
    if not resume_file:
        return None
    if not os.path.exists(resume_file):
        return None
    with open(resume_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    evaluated = payload.get("evaluated_traj_files")
    if isinstance(evaluated, list):
        return set(evaluated)
    return None


def gather_trajectory_files(task: str) -> List[str]:
    pattern = os.path.join(
        "data",
        "json_2.1.0",
        "train",
        f"{task}-*", 
        "trial_*",
        "traj_data.json",
    )
    traj_files = sorted(glob.glob(pattern))
    return traj_files


def run_workers(args: argparse.Namespace, traj_files: Sequence[str],
                logger: EpisodeLogger) -> List[Dict]:
    annotation_limit = max(0, int(args.annotations_per_task))
    gpu_ids = args.gpu_ids or []
    num_workers = max(1, int(args.num_workers))
    num_workers = min(num_workers, len(traj_files)) if traj_files else 1
    aggregated: List[Dict] = []

    def handle_record(record: Dict) -> None:
        logger.log(record)
        aggregated.append(record)

    if args.no_multiprocessing or num_workers == 1:
        gpu_id = gpu_ids[0] if gpu_ids else None
        _process_traj_files(0, traj_files, args, annotation_limit, gpu_id, handle_record)
        return aggregated

    import multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # start method already set

    manager = mp.Manager()
    result_queue = manager.Queue()

    chunks = _chunk_sequence(traj_files, num_workers)
    processes = []
    for worker_id, chunk in enumerate(chunks):
        if not chunk:
            continue
        gpu_id = gpu_ids[worker_id % len(gpu_ids)] if gpu_ids else None
        proc = mp.Process(
            target=_worker_main,
            args=(worker_id, chunk, args, annotation_limit, gpu_id, result_queue),
        )
        proc.daemon = False
        proc.start()
        processes.append(proc)

    finished = 0
    total_processes = len(processes)
    while finished < total_processes:
        try:
            payload = result_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if isinstance(payload, tuple) and payload and payload[0] == "done":
            finished += 1
            continue
        if payload is not None:
            handle_record(payload)

    for proc in processes:
        proc.join()

    return aggregated

def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = f"logs/{args.task}_train_{time.strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(args.output_dir, exist_ok=True)

    traj_files = gather_trajectory_files(args.task)
    if args.limit is not None:
        traj_files = traj_files[: args.limit]

    resume_set = load_resume_set(args.resume)
    if resume_set:
        traj_files = [tf for tf in traj_files if tf not in resume_set]

    if not traj_files:
        print("No pick_and_place_simple trajectories found to evaluate.")
        return

    requested_workers = max(1, int(args.num_workers))
    effective_workers = 1 if args.no_multiprocessing else min(requested_workers, len(traj_files)) or 1
    print(f"Found {len(traj_files)} pick_and_place_simple trajectories in train split.")
    print(f"Using {effective_workers} workers.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logger = EpisodeLogger(args.output_dir, timestamp)

    start_time = time.time()
    try:
        raw_records = run_workers(args, traj_files, logger)
    finally:
        logger.close()
    records = [EpisodeRecord(record) for record in raw_records]

    total_duration = time.time() - start_time
    summary = _summarize(records)
    summary["wall_time_sec"] = total_duration
    summary["evaluated_traj_files"] = len(traj_files)

    breakdown_scene = _breakdown(records, "scene_id")
    breakdown_object = _breakdown(records, "object_target")
    breakdown_parent = _breakdown(records, "parent_target")

    summary_path = os.path.join(args.output_dir, f"summary_{timestamp}.json")
    records_path = logger.json_path
    breakdown_path = os.path.join(args.output_dir, f"breakdowns_{timestamp}.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(records_path, "w", encoding="utf-8") as f:
        json.dump(raw_records, f, indent=2)
    with open(breakdown_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scene": breakdown_scene,
                "object_target": breakdown_object,
                "parent_target": breakdown_parent,
            },
            f,
            indent=2,
        )

    print(f"Summary written to {summary_path}")
    print(f"Per-episode records written to {records_path}")
    print(f"Breakdowns written to {breakdown_path}")
    print(f"Streaming log appended at {logger.jsonl_path}")


if __name__ == "__main__":
    main()
