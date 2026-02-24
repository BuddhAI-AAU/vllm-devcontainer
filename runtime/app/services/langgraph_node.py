from langgraph.graph import StateGraph
from services.memory_node import postgres_memory_node, MemoryState, prompt_builder_node
from services.llm_node import llm_node

builder = StateGraph(MemoryState)

builder.add_node("memory", postgres_memory_node)
builder.add_node("prompt_builder", prompt_builder_node)
builder.add_node("llm", llm_node)

builder.set_entry_point("memory")
builder.add_edge("memory", "prompt_builder")
builder.add_edge("prompt_builder", "llm")
builder.set_finish_point("llm")

graph = builder.compile()


def run_chat(user_id: str, user_text: str):
    result = graph.invoke({
        "user_id": user_id,
        "input": user_text,
        "history": ""
    })
    return result["response"]
