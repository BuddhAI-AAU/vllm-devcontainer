
import requests
from datetime import datetime, timedelta, timezone
import time
import csv

API_URL = "http://localhost:9000/chat"

prompts = [
    "Where does the pause symbol come from",
    "When were the twin towers constructed and deconstructed?",
    "Who is Marcus Hyttel?",
    "Why is Marcus Hyttel?",
    "Who is Marcus Hyttel resq-ing?",
    "What is a Hyttel?",
    "What is the one piece?",
    "Which is the latest Daft Punk song",
    "Is pee stored in the balls?",
    "why is pee stored in the balls",
    "can you list all the fast and furiuos movies?",
    "what would be the environmental impact if snails suddenly dissapeared?",
    "is water wet?",
    "do cows dream?",
    "can spiders hold their breath?",
    "why did Daft Punk stop making music?",
    "summarize the plot of Inland Empire",
    "is curling a real sport?",
    "who actually like watching darts?",
    "can a poor person play golf?",
    "would a vampire be able to go out at night if the moon reflects sunlight?",
    "where does the easter bunny come from and what does he have to do with jesus?",
    "If Alexander Graham Bell invented the telephone, why are telephones not called bells?",
    "When were bells invented?",
    "What is the difference between: Gyros, Shawarma, Döner, kebab, Durum and pita?"
]

def prompt_spam(message: str):
    time_stamp = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

    payload = {
        "user_id": "user123",
        "time_stamp": time_stamp,
        "message": message
    }

    IT_start = time.time()
    response = requests.post(API_URL, json=payload)
    IT_end = time.time()

    inference_time = round(IT_end - IT_start, 3)

    data = response.json()
    model_response = data["response"]
    char_count = len(model_response)

    print(f"\n>>> Prompt: {message}")
    print("Inference time:", inference_time, "seconds")
    print("Model:", model_response)

    return message, model_response, inference_time, char_count


# --- CSV writing section ---
with open("/workspace/csv/results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["prompt", "response", "inference_time", "character_count"])

    for p in prompts:
        row = prompt_spam(p)
        writer.writerow(row)

print("\nCSV saved as results.csv")
