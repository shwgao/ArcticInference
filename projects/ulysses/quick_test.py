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

import time
import vllm
from vllm import LLM, SamplingParams

# Load Ulysses plugins
vllm.plugins.load_general_plugins()

print("Testing Ulysses Configuration...")
print("=" * 50)

# Your configuration
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
    ulysses_sequence_parallel_size=2,
    shift_parallel_threshold=64,
    enable_shift_parallel=True,
    enforce_eager=True,
    max_num_seqs=64,
)

print("LLM initialized successfully!")
print(f"Model: {llm.llm_engine.model_config.model}")
print(f"Max model length: {llm.llm_engine.model_config.max_model_len}")
print(f"Ulysses sequence parallel size: {llm.llm_engine.model_config.ulysses_sequence_parallel_size}")

# Test conversation
conversation = [
    {
        "role": "user",
        "content": "Hello, how are you?"
    }
]

sampling_params = SamplingParams(temperature=0.1, max_tokens=50)

print("\nRunning quick latency test...")
print("-" * 30)

# Warmup
print("Warmup...")
for i in range(3):
    start_time = time.perf_counter()
    outputs = llm.generate([conversation], sampling_params=sampling_params)
    end_time = time.perf_counter()
    if i == 2:  # Last warmup iteration
        warmup_latency = end_time - start_time
        print(f"Warmup latency: {warmup_latency:.4f} seconds")

# Actual test
print("Running test...")
latencies = []
for i in range(5):
    start_time = time.perf_counter()
    outputs = llm.generate([conversation], sampling_params=sampling_params)
    end_time = time.perf_counter()
    latency = end_time - start_time
    latencies.append(latency)
    print(f"Test {i+1}: {latency:.4f} seconds")

# Results
avg_latency = sum(latencies) / len(latencies)
min_latency = min(latencies)
max_latency = max(latencies)

print("\n" + "=" * 50)
print("QUICK TEST RESULTS")
print("=" * 50)
print(f"Average latency: {avg_latency:.4f} seconds")
print(f"Min latency: {min_latency:.4f} seconds")
print(f"Max latency: {max_latency:.4f} seconds")
print(f"Throughput: {50/avg_latency:.2f} tokens/second")

# Show generated text
print(f"\nGenerated text: {outputs[0].outputs[0].text}")

print("\nConfiguration test completed successfully!")
print("Run 'python test_ulysses_latency.py' for comprehensive benchmarking.")
