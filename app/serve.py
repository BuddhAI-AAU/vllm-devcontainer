import subprocess

command = [
    "vllm", "serve", 
    "--model", "mistralai/Mistral-7B-Instruct-v0.3",
    "--hf-token", "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",
    "--max-model-len", "2048",
    "--gpu-memory-utilization", "0.7",
    "--tensor-parallel-size", "2",
    #"--quantization", "Q6_K",
    #"--swap-space", "0",½
    #"--enable-chunked-prefill",
    #"--max-num-batched-tokens", "1024",
    #"--kv-cache-dtype", "fp8",
    "--port", "8000"
]

subprocess.run(command)