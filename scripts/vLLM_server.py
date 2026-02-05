import subprocess

model = "mistralai/Mistral-7B-Instruct-v0.3"
hf_token = "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"
max_model_len = 2048
gpu_memory_utilization = 0.7
tensor_parallel_size = 2
port = 8000
gpu_memory_utilization = 0.7
kv_cache_dtype = "fp8"
quantization = "Q6_K"
max_num_batched_tokens = 1024
enable_chunked_prefill = True


command = [
    "vllm", "serve", 
    "--model", str(model),
    "--hf-token", str(hf_token),
    "--max-model-len", str(max_model_len),
    "--gpu-memory-utilization", str(gpu_memory_utilization),
    "--tensor-parallel-size", str(tensor_parallel_size),
    #"--quantization", "Q6_K",
    #"--swap-space", "0",½
    #"--enable-chunked-prefill",
    #"--max-num-batched-tokens", "1024",
    #"--kv-cache-dtype", "fp8",
    "--port", str(port)
]

subprocess.run(command)