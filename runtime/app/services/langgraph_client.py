import requests
from datetime import datetime, timedelta, timezone
import time
import sounddevice


#server.py API
API_URL = "http://localhost:9000/chat"

#terminal loop
while True:
    user_text = input("You: ")
    time_stamp = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    if not user_text.strip():
        break


    user_id = "lynch"
    MODEL = "mistralai/Ministral-3-14B-Reasoning-2512"  #model vLLM should load. From Hugging Face

    SYSTEM_PROMPT =    ("You are a tutor. Your name is BuddhAi. Base your personality on Buddha" 
                    "Use gentle Socratic questioning to guide the student, but keep answers clear and factual." 
                    "With every question, give specific answers that are satisfying, then proceed with socratic questioning"
                    "Do not repeat the user's question. Do not roleplay. Do not output XML tags.")

    #paramterts sent to vLLM
    params = {
        "model": MODEL,
        "max_tokens": 500,
        "temperature": 0.7,
        "stream": True
    }

    #payload we pass to the server.py
    payload = {
        "user_id": user_id,
        "time_stamp": time_stamp,
        "message": user_text,
        "system_prompt": SYSTEM_PROMPT,
        "params": params
    }

    #inference time start
    IT_start = time.time()
    
    response = requests.post(API_URL, json=payload)

    #inference time end
    IT_end = time.time()

    #prevents loop from breaking if we get a non-200 status
    if response.status_code != 200:             
        print("Server error:", response.text)
        continue


    print("Inference time:", round(IT_end - IT_start, 3), "seconds")
    print()
    print()
    data = response.json()
    print("Model:", data["response"])

    # Play audio if available
    if data.get("audio_base64"):
        import base64, io, sounddevice as sd, soundfile as sf

        audio_bytes = base64.b64decode(data["audio_base64"])
        audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        print("[Playing audio...]")
        sd.play(audio_array, sr)
        sd.wait()
    else:
        print("[No audio]")

    print(time_stamp)