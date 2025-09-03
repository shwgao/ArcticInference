MODEL="meta-llama/Llama-3.1-8B-Instruct" 
TP=2 
SP=1
DP=1
MAX_MODEL_LEN=128000 
OUTPUT_LEN=1 
RESULTS_FILE="benchmark_serve_2tp_1sp_1dp_1node_2gpus.csv" 
OUTPUT_FILE="benchmark_serve_2tp_1sp_1dp_1node_2gpus.txt" 

chunk_prefill_list=(8192)
input_len_list=(1000 5000 10000 40000 50000 80000 100000 120000)
request_rate_list=(1 4 8 12 16 20)

# 1. benchmarks serving
# server
vllm serve gradientai/Llama-3-8B-Instruct-Gradient-1048k --disable-log-requests --tensor_parallel_size 1 --max_model_len 128000
# client
python benchmarks/benchmark_serving.py     --backend vllm     --model gradientai/Llama-3-8B-Instruct-Gradient-1048k     --dataset-name random     --dataset-path None     --random_input_len 32000 --random_output_len 128    --request-rate 1     --num-prompts 20
