models=(openai/gpt-5 anthropic/claude-sonnet-4 google/gemini-2.5-flash deepseek/deepseek-chat-v3.1)
for model in "${models[@]}"; do
  python scripts/run_eval_astar_parallel.py \
    --data-root data/json_2.1.0/train \
    --pattern '' \
    --ridx 0 0 0 0 0 \
    --llm_model "$model" \
    --workers 10 \
    --debug
done