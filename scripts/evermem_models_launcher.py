import subprocess

models = [
    ("Qwen/Qwen3-Embedding-4B", 18000),
    ("BAAI/bge-reranker-v2-m3", 12000),
    #("mistralai/Ministral-3-14B-Reasoning-2512", 9000)
]

processes = []

for model, port in models:
    p = subprocess.Popen([
        "vllm", "serve",
        model,
        "--port", str(port),
        "--gpu-memory-utilization", "0.1",
        "--max-model-len", "1024"
    ])
    processes.append(p)

#Keep script alive
for p in processes:
    p.wait()
