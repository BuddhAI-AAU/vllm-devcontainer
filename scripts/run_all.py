import subprocess
import time
import socket
import sys
import os


# ---------------------------
# Helper: wait for a port
# ---------------------------
def wait_for_port(host, port, name):
    print(f"Waiting for {name} on {host}:{port} ...")
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                print(f"{name} is ready.")
                return
        except OSError:
            time.sleep(0.5)


# ---------------------------
# 1. Start Redis
# ---------------------------
def start_redis():
    print("Starting Redis...")
    try:
        redis_proc = subprocess.Popen(["redis-server"])
    except FileNotFoundError:
        print("ERROR: redis-server not found.")
        print("Install Redis with:")
        print("  sudo apt install redis-server -y")
        sys.exit(1)

    wait_for_port("localhost", 6379, "Redis")
    return redis_proc


# ---------------------------
# 2. Start vLLM
# ---------------------------
def start_vllm():
    print("Starting vLLM...")
    vllm_proc = subprocess.Popen([
        "vllm", "serve",
        "nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4",
        "--port", "8000"
    ])

    wait_for_port("localhost", 8000, "vLLM")
    return vllm_proc


# ---------------------------
# 3. Start FastAPI Gateway
# ---------------------------
def start_fastapi():
    print("Starting FastAPI gateway...")
    fastapi_proc = subprocess.Popen([
        "uvicorn", "main:app",
        "--host", "0.0.0.0",
        "--port", "9000",
        "--reload"
    ])

    wait_for_port("localhost", 9000, "FastAPI Gateway")
    return fastapi_proc


# ---------------------------
# 4. Run LangGraph Memory Agent
# ---------------------------
def run_langgraph_agent():
    print("Running LangGraph memory agent...")
    subprocess.run(["python", "langchain_memory_agent.py"])


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    print("Launching full stack...")

    redis_proc = start_redis()
    vllm_proc = start_vllm()
    fastapi_proc = start_fastapi()

    print("\nAll services are up. Starting LangGraph agent...\n")
    run_langgraph_agent()

    print("Shutting down services...")
    redis_proc.terminate()
    vllm_proc.terminate()
    fastapi_proc.terminate()
