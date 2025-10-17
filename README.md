# Start an OpenAI-compatible Server

## vLLM

```bash
bash servers/vllm_server.sh &> out.txt
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
python vllm_bench.py --model_path llama-8.03b --tokenizer_path meta-llama/Llama-3.1-8B
```

## EasyDeL

```bash
python vllm_bench.py --model_path llama-8.03b --tokenizer_path meta-llama/Llama-3.1-8B
```
