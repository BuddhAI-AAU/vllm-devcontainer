import httpx
import uuid
from datetime import datetime

class MemoryClient:
    def __init__(self, base_url="http://evermem-api:1995", api_key=None, timeout=30):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.Client(
            timeout=timeout,
            headers=self._build_headers(),
        )

    def _build_headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def health(self):
        resp = self._client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def retrieve_memory(self, user_id: str, query: str, top_k: int = 5):
        params = {
            "user_id": user_id,
            "query": query,
            "top_k": top_k,
            "retrieve_method": "hybrid",
        }
        resp = self._client.get(f"{self.base_url}/api/v1/memories/search", params=params)
        resp.raise_for_status()
        return resp.json()

    def write_memory(self, user_id: str, text: str, metadata=None):
        payload = {
            "message_id": str(uuid.uuid4()),
            "create_time": datetime.utcnow().isoformat(),
            "sender": user_id,
            "content": text,
            "role": "assistant",
            "refer_list": [],
        }
        resp = self._client.post(f"{self.base_url}/api/v1/memories", json=payload)
        resp.raise_for_status()
        return resp.json()

    def delete_memory(self, user_id: str):
        payload = {"user_id": user_id}
        resp = self._client.delete(f"{self.base_url}/api/v1/memories", json=payload)
        resp.raise_for_status()
        return resp.json()
