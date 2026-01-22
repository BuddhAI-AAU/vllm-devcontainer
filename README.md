# vllm-devcontainer



vllm-devcontainer/
│
├── .devcontainer/
│   ├── devcontainer.json
│   └── Containerfile          # Dev environment
│
├── app/
│   ├── main.py
│   ├── requirements.txt
│   └── Containerfile          # Runtime container
│
├── pod/
│   └── pod.yaml               # Podman pod definition (optional)
│
├── db/
│   └── init.sql               # Optional DB bootstrap
│
├── scripts/
│   ├── build.sh               # Build runtime containers
│   ├── run.sh                 # Start pod + containers
│   └── stop.sh                # Stop pod + containers
│
└── README.md
