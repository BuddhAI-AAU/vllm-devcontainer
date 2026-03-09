import requests
from datetime import datetime, timedelta, timezone
import time


#server.py API
API_URL = "http://localhost:9000/chat"

#terminal loop
while True:
    user_text = input("You: ")
    time_stamp = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    if not user_text.strip():
        break

#payload we pass to the server.py
    payload = {
        "user_id": "user123",
        "time_stamp": time_stamp,
        "message": user_text
    }

    #inference time start
    IT_start = time.time()
    
    response = requests.post(API_URL, json=payload)

    #inference time end
    IT_end = time.time()

    print("Inference time:", round(IT_end - IT_start, 3), "seconds")
    print()
    print()
    data = response.json()
    print("Model:", data["response"])
    print(time_stamp)