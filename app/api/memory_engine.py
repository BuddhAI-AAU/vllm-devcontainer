from dataclasses import dataclass, field
from typing import List, Callable


#State object
@dataclass
class ConversationState:
    summary: str = ""
    messages: List[str] = field(default_factory=list)


#Summarizer wrapper
class SummarizerLLM:
    def __init__(self, client_fn: Callable[[str], str]):
        self.client_fn = client_fn

    def summarize(self, text: str) -> str:
        prompt = (
            "Summarize the following conversation in a concise way "
            "while preserving important details:\n\n"
            f"{text}\n\nSummary:"
        )
        return self.client_fn(prompt)


class ConversationReducer:
    def __init__(self, summarizer: SummarizerLLM, max_messages: int = 6): #messages to keep before summarizing 
        self.summarizer = summarizer
        self.max_messages = max_messages

    def reduce(self, state: ConversationState, user: str, assistant: str) -> ConversationState: #recieves summary + message buffer, new user message, new assistant message, then returns updated state
        # Add new messages
        state.messages.append(f"User: {user}")
        state.messages.append(f"Assistant: {assistant}")

        # If too many messages, summarize
        if len(state.messages) > self.max_messages:
            joined = "\n".join(state.messages)              #combines messages into single block of text
            new_summary = self.summarizer.summarize(joined)
            state.summary = new_summary                        # stores new summary in the state, making it long-term-memory (LTM)
            state.messages = []  # clear buffer

        return state                                        #returns updated state object


class MemoryEngine:
    def __init__(self, client_fn: Callable[[str], str], max_messages: int = 6):
        self.state = ConversationState()                    #new fresh state object as a memory starting point
        self.reducer = ConversationReducer(
            summarizer=SummarizerLLM(client_fn),
            max_messages=max_messages
        )

    def save(self, user_text: str, assistant_text: str):                        #is called after every user/assistant exchange
        self.state = self.reducer.reduce(self.state, user_text, assistant_text) #passes current state + new messanges to reducer, the reducer then appends msg, summarizes if need be and returns updated state. Then stores the new state

    def get_context(self) -> str:                                               #returns the memory in a format that can be injected into our prompt
        parts = []
        if self.state.summary:                                                  #checks if summary exists, then add it to context LTM
            parts.append(f"Summary:\n{self.state.summary}")
        if self.state.messages:                                                 #add unsummarized messages as "fresh messages", short term memory (STM)
            parts.append("Fresh messages:\n" + "\n".join(self.state.messages))
        return "\n\n".join(parts)                                               #combine all into single string for later injection into prompt
