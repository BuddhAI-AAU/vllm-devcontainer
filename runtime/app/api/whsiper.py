from fastapi import FastAPI, UploadFile, File
import whisper
import uvicorn
import tempfile

app = FastAPI(title="Whisper STT Service")

# Load Whisper ONCE
model = whisper.load_model("base.en")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Save incoming audio to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Run Whisper   
    result = model.transcribe(tmp_path)
    text = result["text"]

    return {"text": text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7001)
