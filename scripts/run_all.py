import subprocess
import socket
import os

def is_port_in_use(port=8000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0

if __name__ == "__main__":
    if is_port_in_use(8000):
        print("vLLM already running.")
    else:
        print("Starting vLLM...")
        subprocess.Popen(["python", "scripts/vLLM_server.py"])

    print("Starting backend...")
    subprocess.Popen(
    [
        (
            "uvicorn.run('main:app', host='0.0.0.0', port=9000)"
        )
    ]
)


    print("All services running.")

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Shutting down...")
