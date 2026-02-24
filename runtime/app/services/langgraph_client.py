import requests

API_URL = "http://localhost:9000/chat"

while True:
    user_text = input("You: ")
    if not user_text.strip():
        break

    payload = {
        "user_id": "user123",
        "message": user_text
    }

    response = requests.post(API_URL, json=payload)
    data = response.json()

    print("Model:", data["response"])
