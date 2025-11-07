vllm serve \
    --model meta-llama/Llama-3.1-8B \
    --seed 42 \
    --tokenizer meta-llama/Llama-3.1-8B \
    --max-model-len 2048 \
    --tensor-parallel-size 4
