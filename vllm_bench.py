#!/usr/bin/env python
# Copyright 2025 Marin Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark script that starts a Levanter inference server and runs vLLM benchmark client against it.

Usage:
    python experiments/bench/vllm_bench.py --model_path <path> --num_prompts 100 --request_rate 10
"""

import argparse
import asyncio
import logging
import multiprocessing
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark."""

    # Model configuration
    model_path: str

    # Server configuration
    host: str
    port: int

    # Benchmark configuration
    num_prompts: int
    request_rate: float
    dataset_name: str
    seed: int
    num_fewshot: int

    tokenizer_path: str | None = None
    dataset_path: str | None = None


async def wait_for_server(host: str, port: int, timeout: int = 60):
    """Wait for the server to be ready."""
    import aiohttp

    url = f"http://{host}:{port}/health"
    start_time = time.time()

    logger.info(f"Waiting for server to be ready at {url}")

    while time.time() - start_time < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                    if response.status == 200:
                        logger.info("Server is ready!")
                        return True
        except Exception as e:
            logger.debug(f"Server not ready yet: {e}")

        await asyncio.sleep(1)

    raise TimeoutError(f"Server did not become ready within {timeout} seconds")


async def run_benchmark_client(config: BenchmarkConfig):
    """Run the vLLM benchmark client."""
    logger.info("Starting benchmark client...")

    # Import the vLLM serve benchmark module
    from vllm_serve import benchmark

    # Prepare benchmark arguments
    api_url = f"http://{config.host}:{config.port}/v1/completions"
    base_url = f"http://{config.host}:{config.port}"

    # Load tokenizer for the benchmark
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_path or config.model_path,
        trust_remote_code=True,
    )

    # Create sample requests
    from datasets import get_samples

    # Create an args object for get_samples
    class SampleArgs:
        def __init__(self):
            self.dataset_name = config.dataset_name
            self.dataset_path = config.dataset_path
            self.num_prompts = config.num_prompts
            self.seed = config.seed
            self.request_id_prefix = "benchmark-serving"
            # Set defaults for optional args that get_samples might check
            self.disable_shuffle = False
            self.backend = "openai"
            self.skip_chat_template = False
            self.no_oversample = False
            # Random dataset specific args
            self.random_prefix_len = 0
            self.random_input_len = 1024
            self.random_output_len = 1024
            self.random_range_ratio = 0.0
            self.random_batch_size = 1

    args = SampleArgs()
    input_requests = get_samples(args, tokenizer)

    logger.info(f"Running benchmark with {len(input_requests)} requests")

    # Run the benchmark
    from vllm_serve import TaskType

    await benchmark(
        task_type=TaskType.GENERATION,
        endpoint_type="openai-nonstreaming",  # Use non-streaming handler for Levanter
        api_url=api_url,
        base_url=base_url,
        model_id=config.model_path,
        model_name=config.model_path,
        tokenizer=tokenizer,
        input_requests=input_requests,
        logprobs=None,
        request_rate=config.request_rate,
        burstiness=1.0,
        disable_tqdm=False,
        profile=False,
        selected_percentile_metrics=["e2el"],  # Only e2el makes sense for non-streaming
        selected_percentiles=[25, 50, 75, 90, 95, 99],
        ignore_eos=True,
        goodput_config_dict={},
        max_concurrency=None,
        extra_headers=None,
        extra_body=None,
        ready_check_timeout_sec=0,  # We already waited for the server
    )

    logger.info("Benchmark completed!")


async def main_async(config: BenchmarkConfig):
    """Main async entry point."""
    # Wait for server to be ready
    await wait_for_server(config.host, config.port)

    # Run the benchmark
    await run_benchmark_client(config)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark Levanter inference server with vLLM client")

    # Model arguments
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B", help="Model path or HF repo")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer path (defaults to model_path)")

    # Server arguments
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")

    # Benchmark arguments
    parser.add_argument("--num_prompts", type=int, default=1000, help="Number of prompts to benchmark")
    parser.add_argument("--request_rate", type=float, default=float("inf"), help="Request rate (requests/sec)")
    parser.add_argument("--dataset_name", type=str, default="random", help="Dataset name")
    parser.add_argument("--dataset_path", type=str, default=None, help="Dataset path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples")

    args = parser.parse_args()

    # Create config
    config = BenchmarkConfig(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        host=args.host,
        port=args.port,
        num_prompts=args.num_prompts,
        request_rate=args.request_rate,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        seed=args.seed,
        num_fewshot=args.num_fewshot,
    )

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the benchmark
    asyncio.run(main_async(config))


if __name__ == "__main__":
    main()
