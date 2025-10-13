import easydel as ed
from transformers import AutoTokenizer
import jax.numpy as jnp

# Initialize model
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    dtype=jnp.float16,
    platform=ed.EasyDeLPlatforms.JAX,
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, -1, 1)
)

# Setup tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-instruct")
tokenizer.pad_token_id = tokenizer.eos_token_id

# Create inference engine
sampling_params=ed.SamplingParams(
    max_tokens=1024,
    temperature=0.8,
    top_p=0.95,
    top_k=10,
)
inference = ed.vInference(
    model=model,
    processor_class=tokenizer,
    generation_config=ed.vInferenceConfig(
        max_new_tokens=1024,
        sampling_params=sampling_params,
        streaming_chunks=32,
    )
)

# Create API server (OpenAI compatible)
api_server = ed.vInferenceApiServer({inference.inference_name: inference})
api_server.fire()
