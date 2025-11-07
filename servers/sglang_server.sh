JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache uv run python -u -m sgl_jax.launch_server \
    --model-path /home/kevin/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/ \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port 8000 \
    --nnodes=1 \
    --context-length=2054 \
    --page-size=16 \
    --tp-size=4 \
    --device=tpu \
    --random-seed=3 \
    --node-rank=0 \
    --mem-fraction-static=0.8 \
    --dtype=bfloat16

# JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache uv run python -m sgl_jax.bench_offline_throughput \
#     --model /home/kevin/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/ \
#     --dataset-name random \
#     --num-prompts 100 \
#     --random-input 1024 \
#     --random-output 1024 \
#     --random-range-ratio 1
