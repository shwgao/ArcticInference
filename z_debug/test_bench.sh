#!/bin/bash

# 简化的测试脚本
MODEL="meta-llama/Llama-3.1-8B-Instruct"
TP=2
PP=1
MAX_MODEL_LEN=131072
OUTPUT_LEN=1

# 设置vllm路径
VLLM_PATH="/nfs/hpc/share/gaosho/conda_envs/arctic-inference/bin/vllm"

# 检查vllm是否可用
if [ ! -f "$VLLM_PATH" ]; then
    echo "Error: vllm not found at $VLLM_PATH"
    exit 1
fi

echo "Using vllm at: $VLLM_PATH"
echo "Model: $MODEL"
echo "Tensor Parallel Size: $TP"
echo "Pipeline Parallel Size: $PP"

# 只测试一个简单的配置
echo "=== Running single test ==="
echo "Testing: input_len=1000, batch_size=1, chunk_prefill=2048"

start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Start time: $start_time"

# 运行vllm命令，添加超时
output=$(timeout 300 $VLLM_PATH bench latency \
  --model $MODEL \
  --max-model-len $MAX_MODEL_LEN \
  --tensor-parallel-size $TP \
  --pipeline-parallel-size $PP \
  --input-len 1000 \
  --output-len $OUTPUT_LEN \
  --trust-remote-code \
  --enforce-eager \
  --load-format dummy \
  --distributed-executor-backend ray \
  --batch-size 1 \
  --num-iters-warmup 2 \
  --num-iters 5 \
  --max-num-batched-tokens 2048 2>&1)

exit_code=$?
end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "End time: $end_time"

if [ $exit_code -eq 124 ]; then
    echo "Warning: Command timed out after 5 minutes"
    echo "Output: $output"
elif [ $exit_code -ne 0 ]; then
    echo "Warning: Command failed with exit code $exit_code"
    echo "Output: $output"
else
    echo "Command completed successfully"
    echo "Output: $output"
fi

# 清理
ray stop

echo "Test finished."
