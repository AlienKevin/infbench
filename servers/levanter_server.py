import equinox as eqx
import haliax as hax
import jax.random as jrandom
import jmp
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.models.lm_model import LmConfig
from levanter.models.llama import LlamaConfig
from levanter.trainer import TrainerConfig
from transformers import AutoConfig

parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length")
parser.add_argument("--max_seqs", type=int, default=256, help="Maximum concurrent sequences")
parser.add_argument("--page_size", type=int, default=128, help="Page size for KV cache")
parser.add_argument("--max_pages", type=int, default=None, help="Maximum number of pages")


def load_model(config):
    """Load a model from HuggingFace checkpoint or local path."""
    tokenizer_path = config.tokenizer_path or config.model_path
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = len(tokenizer)

    mp = config.trainer.mp
    key = jrandom.PRNGKey(config.seed)

    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), config.trainer.compute_axis_mapping)

        # Try loading as HF checkpoint first
        logger.info(f"Loading model from {config.model_path}")
        logger.info(f"Model type: {config.model_config.model_type}")
        logger.info(f"Vocab size: {vocab_size}")
        logger.info(f"Compute dtype: {mp.compute_dtype}")

        try:
            # Check if it's a local Levanter checkpoint
            checkpoint_path = Path(config.model_path)
            if checkpoint_path.exists() and (checkpoint_path / "model").exists():
                logger.info("Loading from Levanter checkpoint")
                model = eqx.filter_eval_shape(config.model_config.build, Vocab, key=key)
                model = load_checkpoint(model, config.model_path, subpath="model")
                model = mp.cast_to_compute(model)
                return model, tokenizer
        except Exception as e:
            logger.debug(f"Not a Levanter checkpoint: {e}")

        # Load from HuggingFace
        logger.info(f"Loading from HuggingFace checkpoint: {config.model_path}")
        converter = HFCheckpointConverter(
            type(config.model_config),
            reference_checkpoint=config.model_path,
            tokenizer=tokenizer,
        )

        model = converter.load_pretrained(
            config.model_config.model_type,
            ref=config.model_path,
            dtype=mp.compute_dtype,
            axis_mapping=config.trainer.parameter_axis_mapping,
        )

        return model, tokenizer


def start_server(config):
    """Start the Levanter inference server in a separate process."""
    logger.info("Starting Levanter inference server...")

    # Load model and create server within the device mesh and axis mapping context
    model, tokenizer = load_model(config)

    server_config = InferenceServerConfig(
        service=InferenceEngineConfig(
            max_seq_len=config.max_seq_len,
            max_seqs=config.max_seqs,
            page_size=config.page_size,
            max_pages=config.max_pages,
        ),
        host=config.host,
        port=config.port,
        temperature=0.7,
        seed=config.seed,
        trainer=config.trainer,
    )

    # Create server within the device mesh and axis mapping context
    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        server = InferenceServer.create(server_config, model, tokenizer)

    logger.info(f"Server initialized, listening on {config.host}:{config.port}")

    # Start serving (blocking call)
    server.serve()


# Trainer configuration
trainer: TrainerConfig = field(
    default_factory=lambda: TrainerConfig(
        model_axis_size=4,
        tensor_parallel_axes=["mlp", "heads", "kv_head", "vocab"],
        mp=jmp.get_policy("p=f32,c=bfloat16"),
    )
)

self.model_config = LlamaConfig.from_hf_config(
            AutoConfig.from_pretrained(self.model_path)
        )
