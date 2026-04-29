from langgraph.graph import StateGraph
from services.memory_node import postgres_memory_node, MemoryState, prompt_builder_node, write_longterm_node, retrieve_longterm_node,MemoryClient, init_memory_client_node
from services.llm_node import llm_node, write_turn_node
from services.longterm_mem import MemoryClient

builder = StateGraph(MemoryState)

builder.add_node("memory", postgres_memory_node)
builder.add_node("prompt_builder", prompt_builder_node)
builder.add_node("llm", llm_node)
#builder.add_node("write_longterm", write_longterm_node) #EverMemOS
#builder.add_node("retrieve_longterm",retrieve_longterm_node ) #EverMemOS
builder.add_node("write_turn", write_turn_node)

builder.set_entry_point("memory")
#builder.add_edge("memory", "retrieve_longterm")         #EverMemOS
#builder.add_edge("retrieve_longterm", "prompt_builder") #EverMemOS
builder.add_edge("memory", "prompt_builder")
builder.add_edge("prompt_builder", "llm")
#builder.add_edge("llm", "write_longterm")
#builder.set_finish_point("write_longterm")
builder.add_edge("llm", "write_turn")
#builder.set_finish_point("llm")
builder.set_finish_point("write_turn")

"""""
builder.add_node("init_memory", init_memory_client_node)
builder.add_node("prompt_builder", prompt_builder_node)
builder.add_node("llm", llm_node)
builder.add_node("write_longterm", write_longterm_node)  # EverMemOS
builder.add_node("retrieve_longterm", retrieve_longterm_node)  # EverMemOS

builder.set_entry_point("init_memory")
builder.add_edge("init_memory", "retrieve_longterm")
builder.add_edge("retrieve_longterm", "prompt_builder")
builder.add_edge("prompt_builder", "llm")
builder.add_edge("llm", "write_longterm")
builder.set_finish_point("write_longterm")
"""



graph = builder.compile()
memory_client = MemoryClient()

def run_chat(user_id: str, user_text: str, time_stamp: str, system_prompt: str, params: dict):
    print("\n=== CLIENT → GRAPH INPUT ===")
    print({"user_id": user_id, "input": user_text, "time_stamp": time_stamp, "system_prompt": system_prompt, "params": params})
    
    
    result = graph.invoke({
        "user_id": user_id,
        "input": user_text,
        "time_stamp": time_stamp,
        "system_prompt": system_prompt,
        "params": params,
        "history": [],
        #"memory_client": memory_client #EverMemOS
    })

    print("\n=== GRAPH → CLIENT OUTPUT ===")
    print(result)

    return result["response"]

