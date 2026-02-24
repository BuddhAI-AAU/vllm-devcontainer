import subprocess
import uvicorn

model = "nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4"
hf_token = "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"
port = 8000 
#KV settings
kv_cache_dtype = "fp8"
kv_cache_type = "nvfp8"                                 #type of the KV cache, adjust based on your GPU capabilities and model requirements. NVFP8 can provide better performance on NVIDIA GPUs that support it, while FP16 is more widely supported but may be slower.
gpu_memory_utilization = 0.9                            #set to a value <1 to avoid OOM errors; adjust based on your GPU's total memory and the model size. Inference speed is higher with higher values.
max_model_len = 2048                                    #size of context window, adjust based on your needs and GPU capabilities. More is better for longer conversations but requires more memory. Inference speed is higher with smaller values.
tensor_parallel_size = 1                                #adjust based on your GPU setup; >1 can speed up inference if you have multiple GPUs. Leave it at 1 for the Spark
quantization = "Q6_K"                                   #Precicion, relevant if the model has options, otherwise keep outcommented.Higher bit values improves accuracy but requires more memory. Higher will reduce inference speed.
max_num_batched_tokens = 16384                           #controls how many tokens are processed in parallel during generation; adjust based on your GPU memory and desired latency. Inference speed is higher with higher values.
enable_chunked_prefill = True                           #enables processing the input prompt in smaller chunks, which can reduce memory usage and latency for long prompts
max_num_seqs = 256                                       #controls how many sequences are generated in parallel; adjust based on your GPU memory and desired throughput. Inference speed is higher with higher values.
enforce_eager = False                                   #If False, we will use CUDA graph and eager execution in hybrid for maximal performance and flexibility.

command = [
    "vllm", "serve", 
    "--config", "runtime/config.yaml",
    #"--model", str(model),
    #"--hf-token", str(hf_token),
    #"--max-model-len", str(max_model_len),
    #"--gpu-memory-utilization", str(gpu_memory_utilization),
    #"--tensor-parallel-size", str(tensor_parallel_size),
    #"--quantization", "Q6_K",
    #"--swap-space", "0",
    #"--enable-chunked-prefill", str(enable_chunked_prefill),
    #"--max-num-seqs", str(max_num_seqs),
    #"--max-num-batched-tokens", str(max_num_batched_tokens),
    #"--enforce-eager", str(enforce_eager),
    #"--kv-cache-dtype", kv_cache_dtype,
    #"--kv-cache-type", kv_cache_type,
    #"--port", str(port)
]

subprocess.run(command)
# Start FastAPI 
#uvicorn.run("app.main:app", host="0.0.0.0", port=9000)
