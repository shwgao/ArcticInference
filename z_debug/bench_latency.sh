# MODEL="nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"
MODEL="meta-llama/Llama-3.1-8B-Instruct" 
TP=1 
SP=2 
DP=1
MAX_MODEL_LEN=128000 
OUTPUT_LEN=1 
RESULTS_FILE="benchmark_latency_1tp_2sp_1dp_1node_2gpus.csv" 
OUTPUT_FILE="benchmark_latency_1tp_2sp_1dp_1node_2gpus.txt" 

# 设置vllm路径
VLLM_PATH="/nfs/hpc/share/gaosho/conda_envs/arctic-inference/bin/vllm"

# 参数列表
chunk_prefill_list=(8192)
input_len_list=(1000 5000 10000 40000 50000 80000 100000 120000)
batch_size_list=(1)

export CUDA_VISIBLE_DEVICES=0,1

# 检查vllm是否可用
if [ ! -f "$VLLM_PATH" ]; then
    echo "Error: vllm not found at $VLLM_PATH"
    exit 1
fi

echo "Using vllm at: $VLLM_PATH"
echo "Model: $MODEL"
echo "Tensor Parallel Size: $TP"
echo "Pipeline Parallel Size: $PP"

# 写表头
echo "chunk_prefill,input_len,batch_size,avg_latency,p10_latency,p25_latency,p50_latency,p75_latency,p90_latency,p99_latency" > $RESULTS_FILE

# 计算总测试数量
total_tests=$((${#chunk_prefill_list[@]} * ${#input_len_list[@]} * ${#batch_size_list[@]}))
current_test=0


for chunk_prefill in "${chunk_prefill_list[@]}"; do
  for input_len in "${input_len_list[@]}"; do
    for batch_size in "${batch_size_list[@]}"; do
      current_test=$((current_test + 1))
      echo "=== Test $current_test/$total_tests ==="
      echo "Running: chunk_prefill=$chunk_prefill, input_len=$input_len, batch_size=$batch_size"
      echo "Progress: $current_test/$total_tests ($(($current_test * 100 / $total_tests))%)"
      
      # 添加时间戳
      start_time=$(date +"%Y-%m-%d %H:%M:%S")
      echo "Start time: $start_time"

      CMD="$VLLM_PATH bench latency \
        --model $MODEL \
        --max-model-len $MAX_MODEL_LEN \
        --tensor-parallel-size $TP \
        --data-parallel-size $DP \
        --enable-shift-parallel \
        --shift-parallel-threshold 512 \
        --ulysses_sequence_parallel_size 2 \
        --input-len $input_len \
        --output-len $OUTPUT_LEN \
        --trust-remote-code \
        --enforce-eager \
        --load-format dummy \
        --batch-size $batch_size \
        --num-iters-warmup 2\
        --num-iters 10\
        --max-num-batched-tokens $chunk_prefill 2>&1"

      echo "running command: $CMD"
      
      # 运行vllm命令，添加超时和更好的错误处理
      output=$($CMD 2>&1)
      
      exit_code=$?
      end_time=$(date +"%Y-%m-%d %H:%M:%S")
      echo "End time: $end_time"
      
      if [ $exit_code -eq 124 ]; then
        echo "Warning: Command timed out after 5 minutes"
        output="TIMEOUT: Command exceeded 5 minute limit"
      elif [ $exit_code -ne 0 ]; then
        echo "Warning: Command failed with exit code $exit_code"
      else
        echo "Command completed successfully"
      fi

      # 打印输出调试（取消注释以查看完整输出）
      echo "$output" > $OUTPUT_FILE

      # 提取延迟指标
      avg_latency=$(echo "$output" | grep "Avg latency:" | sed 's/.*Avg latency: \([0-9.]*\) seconds.*/\1/')
      p10_latency=$(echo "$output" | grep "10% percentile latency:" | sed 's/.*10% percentile latency: \([0-9.]*\) seconds.*/\1/')
      p25_latency=$(echo "$output" | grep "25% percentile latency:" | sed 's/.*25% percentile latency: \([0-9.]*\) seconds.*/\1/')
      p50_latency=$(echo "$output" | grep "50% percentile latency:" | sed 's/.*50% percentile latency: \([0-9.]*\) seconds.*/\1/')
      p75_latency=$(echo "$output" | grep "75% percentile latency:" | sed 's/.*75% percentile latency: \([0-9.]*\) seconds.*/\1/')
      p90_latency=$(echo "$output" | grep "90% percentile latency:" | sed 's/.*90% percentile latency: \([0-9.]*\) seconds.*/\1/')
      p99_latency=$(echo "$output" | grep "99% percentile latency:" | sed 's/.*99% percentile latency: \([0-9.]*\) seconds.*/\1/')

      # 检查是否成功提取到延迟数据
      if [ -z "$avg_latency" ] || [ -z "$p10_latency" ] || [ -z "$p25_latency" ] || [ -z "$p50_latency" ] || [ -z "$p75_latency" ] || [ -z "$p90_latency" ] || [ -z "$p99_latency" ]; then
        echo "Warning: Failed to extract latency data, setting to N/A"
        avg_latency="N/A"
        p10_latency="N/A"
        p25_latency="N/A"
        p50_latency="N/A"
        p75_latency="N/A"
        p90_latency="N/A"
        p99_latency="N/A"
      else
        echo "Successfully extracted latency data:"
        echo "  Avg: $avg_latency, P10: $p10_latency, P50: $p50_latency, P99: $p99_latency"
      fi

      echo "$chunk_prefill,$input_len,$batch_size,$avg_latency,$p10_latency,$p25_latency,$p50_latency,$p75_latency,$p90_latency,$p99_latency" >> $RESULTS_FILE
      echo "Results saved to $RESULTS_FILE"
      echo ""
    done
  done
done

echo "All benchmarks done. Results saved to $RESULTS_FILE" 


# sleep 1000000

# --- 6. Clean up ---
ray stop

echo "Job finished."