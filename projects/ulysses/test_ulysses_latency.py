# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import dataclasses
import json
import os
import time
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

import vllm
from vllm import LLM, SamplingParams
from vllm.inputs import PromptType

# Load Ulysses plugins
vllm.plugins.load_general_plugins()


def add_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int,default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of generated sequences per prompt.",
    )
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=10,
        help="Number of iterations to run for warmup.",
    )
    parser.add_argument("--num-iters",
                        type=int,
                        default=30,
                        help="Number of iterations to run.")
    parser.add_argument(
        "--output-json",
        type=str,
        default="ulysses_latency_results.json",
        help="Path to save the latency results in JSON format.",
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=("Do not detokenize responses (i.e. do not include "
              "detokenization time in the latency measurement)"),
    )
    # Ulysses specific parameters
    parser.add_argument("--ulysses-seq-parallel-size", type=int, default=2)
    parser.add_argument("--shift-parallel-threshold", type=int, default=64)
    parser.add_argument("--enable-shift-parallel", action="store_true", default=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=64)


def main(args: argparse.Namespace):
    print("Initializing Ulysses LLM with configuration:")
    print(f"  - Ulysses sequence parallel size: {args.ulysses_seq_parallel_size}")
    print(f"  - Shift parallel threshold: {args.shift_parallel_threshold}")
    print(f"  - Enable shift parallel: {args.enable_shift_parallel}")
    print(f"  - Tensor parallel size: {args.tensor_parallel_size}")
    print(f"  - Max num sequences: {args.max_num_seqs}")
    print(f"  - Input length: {args.input_len}")
    print(f"  - Output length: {args.output_len}")
    print(f"  - Batch size: {args.batch_size}")
    print("=" * 80)

    # Initialize LLM with Ulysses configuration
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=args.tensor_parallel_size,
        ulysses_sequence_parallel_size=args.ulysses_seq_parallel_size,
        shift_parallel_threshold=args.shift_parallel_threshold,
        enable_shift_parallel=args.enable_shift_parallel,
        enforce_eager=True,
        max_num_seqs=args.max_num_seqs,
    )

    # Verify model configuration
    assert llm.llm_engine.model_config.max_model_len >= (
        args.input_len + args.output_len), (
        "Please ensure that max_model_len is greater than "
        "the sum of input_len and output_len.")

    sampling_params = SamplingParams(
        n=args.n,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
        detokenize=not args.disable_detokenize,
    )

    # Create dummy prompts for testing
    dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(args.batch_size,
                                                     args.input_len))
    dummy_prompts: list[PromptType] = [{
        "prompt_token_ids": batch
    } for batch in dummy_prompt_token_ids.tolist()]

    def llm_generate():
        llm.generate(dummy_prompts,
                     sampling_params=sampling_params,
                     use_tqdm=False)

    def run_to_completion():
        start_time = time.perf_counter()
        llm_generate()
        end_time = time.perf_counter()
        latency = end_time - start_time
        return latency

    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion()

    print("Running latency benchmark...")
    # Benchmark
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Benchmark iterations"):
        latencies.append(run_to_completion())
    
    latencies = np.array(latencies)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies, percentages)
    
    print("\n" + "=" * 80)
    print("LATENCY BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Average latency: {np.mean(latencies):.4f} seconds")
    print(f"Min latency: {np.min(latencies):.4f} seconds")
    print(f"Max latency: {np.max(latencies):.4f} seconds")
    print(f"Std deviation: {np.std(latencies):.4f} seconds")
    print("\nPercentiles:")
    for percentage, percentile in zip(percentages, percentiles):
        print(f"  {percentage}%: {percentile:.4f} seconds")

    # Calculate throughput
    total_tokens = args.batch_size * args.output_len
    avg_throughput = total_tokens / np.mean(latencies)
    print(f"\nThroughput: {avg_throughput:.2f} tokens/second")

    # Output JSON results
    if args.output_json:
        results = {
            "configuration": {
                "ulysses_sequence_parallel_size": args.ulysses_seq_parallel_size,
                "shift_parallel_threshold": args.shift_parallel_threshold,
                "enable_shift_parallel": args.enable_shift_parallel,
                "tensor_parallel_size": args.tensor_parallel_size,
                "max_num_seqs": args.max_num_seqs,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "batch_size": args.batch_size,
            },
            "metrics": {
                "avg_latency": float(np.mean(latencies)),
                "min_latency": float(np.min(latencies)),
                "max_latency": float(np.max(latencies)),
                "std_latency": float(np.std(latencies)),
                "throughput_tokens_per_sec": float(avg_throughput),
                "latencies": latencies.tolist(),
                "percentiles": dict(zip([f"{p}%" for p in percentages], percentiles.tolist())),
            }
        }
        
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Ulysses LLM latency")
    add_cli_args(parser)
    args = parser.parse_args()
    main(args)
