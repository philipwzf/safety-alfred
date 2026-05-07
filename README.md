# Safety-ALFRED

LLM-agent safety evaluation on ALFRED tasks in AI2-THOR. The pipeline rolls out an LLM agent in the simulator, records each trajectory, and (i) optionally runs a safety critic in the loop and (ii) scores the trajectory against CTL (computation tree logic) safety constraints post-hoc.

## Quickstart

Install requirements (conda):

```bash
conda create -n ai2thor python==3.10
conda activate ai2thor
pip install -r requirements.txt
export API_KEY="your_openrouter_api_key"
```

Single-trajectory smoke test (any agent mode; details below):

```bash
python models/eval/eval_llm_step.py --debug \
    --traj_file data/json_2.1.0/train/look_at_obj_in_light-AlarmClock-None-DeskLamp-305/trial_T20190908_082723_323728/traj_data.json
```

Parallel sweep over a split:

```bash
bash scripts/run_all.bash
```

CTL safety summary (post-hoc, ground truth):

```bash
python safety_eval/ctl_full_pipeline.py --model-name openai/gpt-5
```

For headless servers, see [Headless Server](#headless-server).

---

## Log Namespace Convention

`--log-model-name` is the directory under `logs/trajectories/` that the rollout writes into and the key consumed by `safety_eval/ctl_full_pipeline.py` for the post-hoc CTL summary. Recommended pattern:

```text
{provider}/{model-id}-{rollout-mode}
```

Examples:

```text
openai/gpt-5
deepseek/deepseek-chat-v3.1
deepseek/v3.1-fullep-deterministic
deepseek/v3.1-fullep-llmjudge
```

The trailing segment lets you keep different rollout regimes (baseline / stepwise-ReAct / full-episode ReAct with each critic type) in separate namespaces under the same provider/model.

---

## Evaluation Modes

The pipeline supports three regimes that differ in **where safety feedback enters the rollout**:

- **Baseline** — no safety critic. The planner generates and the agent executes; safety is judged only post-hoc.
- **Stepwise ReAct** — per-action safety gate. The planner proposes one next action; a deterministic critic checks it before execution; on `unsafe`, one round of feedback drives a re-planned action.
- **Full-Episode ReAct** — trace-level safety loop. The planner generates a complete plan; the agent executes it; the critic evaluates the entire trace; if unsafe (or task didn't succeed), the planner re-plans the entire trajectory from the original initial scene with safety feedback, up to a max-loop cap.

For full-episode ReAct only, two critic types are available via `--critic-type`:

- `deterministic` → SENTINEL rule-based critic; uses the same CTL evaluator as the post-hoc pipeline (in-loop and post-hoc verdicts are identical by construction).
- `llm` → LLM-as-Judge critic; same backbone as the planner; produces an independent in-loop verdict, with CTL run separately as ground truth.

The parallel sweep runner (`scripts/run_eval_astar_parallel.py`) drives any mode by swapping `--eval-script`.

### Baseline

```bash
# Stepwise (one LLM call per action)
python models/eval/eval_llm_step.py --debug \
    --traj_file data/json_2.1.0/train/<task>/<trial>/traj_data.json

# A*-expanded GotoLocation (recommended for navigation-heavy tasks)
python models/eval/eval_llm_astar.py --debug \
    --traj_file data/json_2.1.0/train/<task>/<trial>/traj_data.json
```

Parallel sweep over the full `train` split:

```bash
python scripts/run_eval_astar_parallel.py \
    --data-root data/json_2.1.0/train \
    --pattern '' \
    --eval-script models/eval/eval_llm_astar.py \
    --model deepseek/deepseek-chat-v3.1 \
    --log-model-name deepseek/deepseek-chat-v3.1 \
    --workers 10 --max_steps 50 --max_fails 5 --debug
```

`scripts/run_all.bash` wraps the above for a fixed list of models.

### Stepwise ReAct

Adds the deterministic safety critic on top of the stepwise loop. If the proposed next action is unsafe, the critic gives one round of explicit safety feedback to the planner and the revised action executes.

```bash
python scripts/run_eval_astar_parallel.py \
    --data-root data/json_2.1.0/train \
    --pattern '' \
    --eval-script models/eval/eval_llm_react.py \
    --model deepseek/deepseek-chat-v3.1 \
    --log-model-name deepseek/v3.1-stepwise-react \
    --workers 10 --max_steps 50 --max_fails 5 --debug
```

### Full-Episode ReAct

Wraps the entire rollout in a feedback loop. After each rollout, the trace-level critic evaluates the saved trace; if not Safe & Success and the loop cap is not yet hit, the planner re-plans the **entire trajectory** from the original initial scene with safety feedback (passed to both subgoal generation and action plan generation), then executes the new plan. The loop exits on the first iteration where the critic declares Safe & Success, or when `--max-loop-limit` is reached.

`--max-loop-limit` controls the maximum number of rollout iterations (default `3`). Per-iteration intermediate traces are saved under `logs/intermediate_traces/<log-model-name>/...`.

Run the full `train` split with the deterministic SENTINEL critic:

```bash
python scripts/run_eval_astar_parallel.py \
    --data-root data/json_2.1.0/train \
    --pattern '' \
    --eval-script models/eval/eval_llm_full_react.py \
    --model deepseek/deepseek-chat-v3.1 \
    --log-model-name deepseek/v3.1-fullep-deterministic \
    --critic-type deterministic \
    --max-loop-limit 3 \
    --workers 10 --max_steps 50 --debug
```

Run with the LLM-as-Judge critic:

```bash
python scripts/run_eval_astar_parallel.py \
    --data-root data/json_2.1.0/train \
    --pattern '' \
    --eval-script models/eval/eval_llm_full_react.py \
    --model deepseek/deepseek-chat-v3.1 \
    --log-model-name deepseek/v3.1-fullep-llmjudge \
    --critic-type llm \
    --max-loop-limit 3 \
    --workers 10 --max_steps 50 --debug
```

#### Loop semantics

With `--max-loop-limit 3`, the value `loops_run` recorded under each trace's `full_episode_react.loops_run` field has the following meaning:

| `loops_run` | rollouts performed | feedbacks consumed by the planner |
|---:|---:|---:|
| 1       | 1 (no feedback) | 0 — first rollout reached Safe & Success without any feedback |
| 2       | 2               | 1 — feedback from rollout #1 used to drive rollout #2; rollout #2 hit Safe & Success |
| 3 (cap) | 3               | 2 — feedback from rollout #1 used in rollout #2; feedback from #2 used in #3; loop cap hit |

The final trace (the one CTL summary keys off) is the loop-exit iteration's trace; the intermediate `round_NN.json` files preserve every prior iteration for ablation.

---

## Post-hoc CTL Safety Summary

Run after rollouts finish (any regime, any critic type):

```bash
python safety_eval/ctl_full_pipeline.py \
    --model-name deepseek/v3.1-fullep-deterministic \
    --model-split train \
    --constraints-json safety_rules_object.json
```

Or score a single task across all models:

```bash
python safety_eval/ctl_full_pipeline.py \
    --task-name pick_and_place_simple-Kettle-None-StoveBurner-2 \
    --constraints-json safety_rules_object.json
```

The summary writes `ctl_results_*.json` next to the scored traces and prints aggregate Safe / Success / Safe-and-Success counts.

---

## Headless Server

```bash
# Start Xvfb on display :99
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +extension RANDR +extension RENDER &
export DISPLAY=:99

# Smoke test
python scripts/check_thor.py
  ###############
  ## (300, 300, 3)
  ## Everything works!!!
```

Set `AI2THOR_X_DISPLAY=99` (or update the constant in [`gen/constants.py`](gen/constants.py)) so AI2-THOR controllers find the same display.

Also see: [Setting up THOR on Google Cloud](https://medium.com/@etendue2013/how-to-run-ai2-thor-simulation-fast-with-google-cloud-platform-gcp-c9fcde213a4a)

---

## Citation

If you find the dataset or code useful, please cite:

```
@inproceedings{ALFRED20,
  title ={{ALFRED: A Benchmark for Interpreting Grounded
           Instructions for Everyday Tasks}},
  author={Mohit Shridhar and Jesse Thomason and Daniel Gordon and Yonatan Bisk and
          Winson Han and Roozbeh Mottaghi and Luke Zettlemoyer and Dieter Fox},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020},
  url  = {https://arxiv.org/abs/1912.01734}
}
```

## License

MIT License
