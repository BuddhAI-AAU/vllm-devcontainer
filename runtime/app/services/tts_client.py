from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="not-needed"
)

def synthesize_tts(text: str) -> bytes:
    resp = client.audio.speech.create(
    model="mistralai/Voxtral-4B-TTS-2603",
    voice="casual_male",
    input=text,
    response_format="wav"
)
    return resp.read()
