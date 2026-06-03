#deprecated
import subprocess

cmd = [
    "vllm",
    "serve",
    "mistralai/Voxtral-Mini-4B-Realtime-2602",
    "--omni",
    "--port", "7001",

    # VRAM‑saving flags:
    "--max-model-len", "2048",
    "--gpu-memory-utilization", "0.15",
    "--max-num-seqs", "1",
    "--max-num-batched-tokens", "2048",
    "--enforce-eager",
    #"--dtype", "bf16",
]

subprocess.run(cmd)
