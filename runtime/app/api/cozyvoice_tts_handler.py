import sys
sys.path.insert(0, "/workspace/CosyVoice")
from hyperpyyaml import load_hyperpyyaml


from fastapi import FastAPI
from pydantic import BaseModel
import base64
import torch
from cosyvoice.cli.cosyvoice import AutoModel

app = FastAPI()
cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

class InstructTTSRequest(BaseModel):
    text: str
    instruct: str
    spk_id: str = "default"

@app.post("/tts/instruct")
def tts_instruct(req: InstructTTSRequest):
    audio_iter = cosyvoice.inference_instruct(
        req.text,
        req.spk_id,
        req.instruct,
        stream=False
    )

    chunks = [j["tts_speech"] for _, j in enumerate(audio_iter)]
    audio_tensor = torch.cat(chunks, dim=1)
    audio_bytes = audio_tensor.cpu().numpy().tobytes()

    return {"audio": base64.b64encode(audio_bytes).decode("utf-8")}
