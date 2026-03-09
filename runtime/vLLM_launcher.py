import subprocess

command = [
    "vllm", "serve", 
    "--config", "runtime/config.yaml",
]

subprocess.run(command)

