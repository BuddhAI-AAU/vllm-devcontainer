import httpx
import uuid
from datetime import datetime

longterm_print = False

class MemoryClient:
    def __init__(self, base_url="http://evermem-api:1995", api_key=None, timeout=30, debug =True):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.debug = debug                  #debug log
        self._client = httpx.Client(
            timeout=timeout,
            headers=self._build_headers(),
        )

    #debug log
    def _log(self, *args):
        if self.debug:
            print("Memory Client", *args)

    def _build_headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def health(self):
        url = f"{self.base_url}/health"                         #debug log
        self._log("GET", url)                                   #debug log
        resp = self._client.get(f"{self.base_url}/health")
        self._log("Response:", resp.status_code, resp.text)     #debug log
        resp.raise_for_status()
        return resp.json()

    def retrieve_memory(self, user_id: str, query: str, top_k: int = 5):
        params = {
            "user_id": user_id,
            "query": query,
            "top_k": top_k,
            "retrieve_method": "hybrid",
        }
        url = f"{self.base_url}/api/v1/memories/search"         #debug log
        self._log("GET", url, "params:", params)                #debug log
        resp = self._client.get(f"{self.base_url}/api/v1/memories/search", params=params)
        self._log("Response:", resp.status_code, resp.text)         #debug log
        resp.raise_for_status()
        return resp.json()

    def write_memory(self, user_id: str, text: str, metadata=None):
        payload = {
            "user_id": user_id,
            "message_id": str(uuid.uuid4()),
            "create_time": datetime.utcnow().isoformat(),
            "sender": user_id,
            "content": text,
            "role": "assistant",
            "refer_list": [],
        }
        url = f"{self.base_url}/api/v1/memories"        #debug log
        self._log("POST", url, "payload:", payload)     #debug log
        resp = self._client.post(f"{self.base_url}/api/v1/memories", json=payload)
        self._log("Response:", resp.status_code, resp.text) #debug log
        resp.raise_for_status()
        return resp.json()

    def delete_memory(self, user_id: str):
        payload = {"user_id": user_id}
        url = f"{self.base_url}/api/v1/memories"        #debug log
        self._log("DELETE", url, "payload:", payload)   #debug log
        resp = self._client.delete(f"{self.base_url}/api/v1/memories", json=payload)
        self._log("Response:", resp.status_code, resp.text)     #debug log
        resp.raise_for_status()
        return resp.json()


if __name__ == "__main__":
    client = MemoryClient(debug=True)

    user_id = "oscar"
    messages = [
        "My favorite food is tacos.",
        "My favorite band is zz top",
        "What is a float?"
    ]

    print("=== Writing Memories ===")
    for msg in messages:
        client.write_memory(user_id, msg)

    print("\n=== Retrieving Memories ===")
    retrieved = client.retrieve_memory(user_id, "message")
    print("Retrieved:", retrieved)