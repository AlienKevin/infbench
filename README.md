# Start an OpenAI-compatible Server

## vLLM

Start vLLM docker
```bash
sudo docker run \
      --pull=always \
      -v $SSH_AUTH_SOCK:/ssh-agent \
      -e SSH_AUTH_SOCK=/ssh-agent \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      --privileged \
      -itd \
      --net host \
      --shm-size=16G \
      --hostname vllm-tpu \
      --name vllm-tpu-container \
      --env HUGGING_FACE_HUB_TOKEN=xxx \
      vllm/vllm-tpu:v0.11.1
```

If container is already running, you can exec into it:
```bash
sudo docker exec -it vllm-tpu-container bash
```

Paste content of `servers/vllm_server.sh` into the container and run.

## SGLang-JAX

```bash
git clone https://github.com/sgl-project/sglang-jax.git
cd sglang-jax
uv venv --python 3.12 && source .venv/bin/activate
uv pip install sglang-jax
```

## Levanter

```bash
python servers/levanter_server.py &> out.txt
```

## EasyDeL

```bash
python servers/easydel_server.py &> out.txt
```

# Start a Load Generator Client

## vLLM/Levanter

To work around EasyDeL changing `model_path` internally:

```bash
python vllm_bench.py --model_path meta-llama/Llama-3.1-8B --tokenizer_path meta-llama/Llama-3.1-8B
```

## EasyDeL

```bash
python vllm_bench.py --model_path llama-8.03b --tokenizer_path meta-llama/Llama-3.1-8B
```

# Benchmark results on v4-8

## vLLM

```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  308.06    
Total input tokens:                      1023000   
Total generated tokens:                  1024000   
Request throughput (req/s):              3.25      
Output token throughput (tok/s):         3324.05   
Peak output token throughput (tok/s):    84.00     
Peak concurrent requests:                1000.00   
Total Token throughput (tok/s):          6644.86   
----------------End-to-end Latency----------------
Mean E2EL (ms):                          194384.25 
Median E2EL (ms):                        168641.90 
P25 E2EL (ms):                           90387.84  
P50 E2EL (ms):                           168641.90 
P75 E2EL (ms):                           246410.87 
P90 E2EL (ms):                           306384.01 
P95 E2EL (ms):                           307139.50 
P99 E2EL (ms):                           307487.23 
==================================================
```

## SGLang-JAX

```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  444.73    
Total input tokens:                      1023000   
Total generated tokens:                  1024000   
Request throughput (req/s):              2.25      
Output token throughput (tok/s):         2302.54   
Peak output token throughput (tok/s):    309.00    
Peak concurrent requests:                1000.00   
Total Token throughput (tok/s):          4602.83   
----------------End-to-end Latency----------------
Mean E2EL (ms):                          301166.84 
Median E2EL (ms):                        290417.15 
P25 E2EL (ms):                           152709.74 
P50 E2EL (ms):                           290417.15 
P75 E2EL (ms):                           428203.95 
P90 E2EL (ms):                           428246.84 
P95 E2EL (ms):                           435559.38 
P99 E2EL (ms):                           444237.92 
==================================================
```
