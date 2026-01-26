import subprocess

command = [
    "vllm", "serve", 
    "--model", "allenai/OLMo-1B-hf",
    "--hf-token", "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",
    "--max-model-len", "1024",
    "--gpu-memory-utilization", "0.7",
    #"--tensor-parallel-size", "1",
    #"--swap-space", "0",½
    #"--enable-chunked-prefill",
    #"--max-num-batched-tokens", "1024",
    #"--kv-cache-dtype", "fp8",
    "--port", "8000"
]

subprocess.run(command)