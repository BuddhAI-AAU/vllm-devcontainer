import subprocess

command2 = [
    "vllm", "bench", "serve",
    "--model", "nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4",
    "--backend", "vllm",
    "--host", "127.0.0.1",
    "--port", "8000",
    "--dataset-name", "random",
    "--num-prompts", "2",
    "--request-rate", "2"
]

subprocess.run(command2)