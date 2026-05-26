import subprocess

cmd = [
    "vllm",
    "serve",
    "mistralai/Voxtral-4B-TTS-2603",
    "--omni",
    "--port", "7000",

    # VRAM‑saving flags:
    "--max-model-len", "2048",
    "--gpu-memory-utilization", "0.15",
    "--max-num-seqs", "1",
    "--max-num-batched-tokens", "2048",
    "--enforce-eager",
    #"--dtype", "bf16",
]

subprocess.run(cmd)
