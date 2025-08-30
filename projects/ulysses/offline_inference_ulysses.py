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

import vllm
from vllm import LLM, SamplingParams


vllm.plugins.load_general_plugins()

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
    # pipeline_parallel_size=2,
    ulysses_sequence_parallel_size=2,
    shift_parallel_threshold=64,
    enable_shift_parallel=True,
    enforce_eager=True,
    max_num_seqs=64,
)

print("=" * 80)

def format_conversation_to_prompt(conversation):
    """Convert conversation format to a text prompt that vLLM can understand."""
    prompt = ""
    for message in conversation:
        if message["role"] == "user":
            prompt += f"<|user|>\n{message['content']}\n<|assistant|>\n"
        elif message["role"] == "assistant":
            prompt += f"{message['content']}\n"
    return prompt

conversation = [
    {
        "role": "user",
        "content": "Hello"
    },
    {
        "role": "assistant",
        "content": "Hello! How can I assist you today?"
    },
    {
        "role": "user",
        "content": "Write an essay about the importance of higher education."*100,
    },
]

# Convert conversations to prompts
prompt1 = format_conversation_to_prompt(conversation)
prompt2 = format_conversation_to_prompt(conversation)

sampling_params = SamplingParams(temperature=0.1, max_tokens=800)

outputs = llm.generate([prompt1, prompt2], sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
print(outputs[1].outputs[0].text)
