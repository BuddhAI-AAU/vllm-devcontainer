from langgraph.graph import StateGraph
from runtime.app.services.memory_node import (
    postgres_memory_node,
    MemoryState,
    prompt_builder_node,
)
from runtime.app.services.llm_node import llm_node, write_turn_node
from scripts.perf_logger import log_perf   # <-- CSV logger
import time

def now():
    return time.perf_counter()

# ---------------- TIMED NODE WRAPPER ----------------

def timed_node(name, fn):
    def wrapper(state):
        t0 = now()
        out = fn(state)
        dt = now() - t0

        user_id = state.get("user_id", "unknown")

        print(f"[TIMING] Node {name}: {dt:.3f} sec")
        log_perf(
            user_id=user_id,
            stage=f"node_{name}",
            duration=dt
        )

        return out
    return wrapper

# ---------------- GRAPH SETUP ----------------

builder = StateGraph(MemoryState)

builder.add_node("memory", timed_node("memory", postgres_memory_node))
builder.add_node("prompt_builder", timed_node("prompt_builder", prompt_builder_node))
builder.add_node("llm", timed_node("llm", llm_node))
builder.add_node("write_turn", timed_node("write_turn", write_turn_node))

builder.set_entry_point("memory")
builder.add_edge("memory", "prompt_builder")
builder.add_edge("prompt_builder", "llm")
builder.add_edge("llm", "write_turn")
builder.set_finish_point("write_turn")

graph = builder.compile()

# ---------------- RUN CHAT ----------------

def run_chat(user_id: str, user_text: str, time_stamp: str, system_prompt: str, params: dict):

    print("\n=== CLIENT → GRAPH INPUT ===")
    print({
        "user_id": user_id,
        "input": user_text,
        "time_stamp": time_stamp,
        "system_prompt": system_prompt,
        "params": params
    })

    # ---- FORCE STREAMING ----
    params["stream"] = True

    messages = [{"role": "user", "content": user_text}]

    # ---- TIMING: FULL GRAPH ----
    t0 = now()
    result = graph.invoke({
        "user_id": user_id,
        "input": messages,
        "time_stamp": time_stamp,
        "system_prompt": system_prompt,
        "params": params,
        "history": [],
    })
    t_graph = now() - t0

    print(f"[TIMING] LangGraph.invoke(): {t_graph:.3f} sec")

    # CSV log for full graph
    log_perf(
        user_id=user_id,
        stage="graph_total",
        duration=t_graph
    )

    # ---- Extract output ----
    text = result["response"]

    print("\n=== GRAPH → CLIENT OUTPUT ===")
    print(result)

    # ---- No TTS here (TTS is only in WebSocket streaming path) ----
    return {
        "response": text,
        "audio_base64": None
    }
