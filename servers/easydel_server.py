import time

import easydel as ed

# add in reserve_tokens
max_model_len = 2048 + 800
elarge = ed.eLargeModel(
    {
        "model": {
            "name_or_path": "meta-llama/Llama-3.1-8B",
            "tokenizer": "meta-llama/Llama-3.1-8B",
        },
        "sharding": {
            "axis_dims": (1, 1, 1, -1, 1),
            "axis_names": ("dp", "fsdp", "ep", "tp", "sp"),
            "auto_shard_model": True,
        },
        "esurge": {
            "enable_prefix_caching": True,
            "hbm_utilization": 0.85,
            "max_model_len": max_model_len,
            "max_num_seqs": 512,
            "min_input_pad": 128,
            "page_size": 128,
            "verbose": False,
        },
        "loader": {"dtype": "bf16", "param_dtype": "bf16", "precision": "DEFAULT"},
        "base_config": {
            "values": {
                "mask_max_position_embeddings": max_model_len,
                "freq_max_position_embeddings": max_model_len,
                "attn_mechanism": "auto",
                "use_pallas_group_matmul": False,
                "gradient_checkpointing": "checkpoint_dots",
            }
        },
        "eval": {"apply_chat_template": False, "temperature": 0.0, "max_new_tokens": max_model_len},
    }
)
surge = elarge.build_esurge()
surge.start_monitoring()
# surge.generate(
#     ["something here so u should continue generating things like this non stop :)) " for _ in range(3000)],
#     sampling_params=ed.SamplingParams(max_tokens=1024),
# )
# time.sleep(50)

# Launch API server
server = ed.eSurgeApiServer(surge)
server.fire(host="127.0.0.1", port=8000)
