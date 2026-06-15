import subprocess

command = [
    "vllm", "serve", 
    "--config", "runtime/app/api/config.yaml",
]

subprocess.run(command)

