import torch
import base64
import numpy as np
from omnivoice import OmniVoice

DEFAULT_INSTRUCT = "male"

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16
)

def run_tts(text: str, instruct: str | None = None) -> str:
    instruct = instruct or DEFAULT_INSTRUCT

    audio = model.generate(
        text=text,
        instruct=instruct,
    )

    # -----------------------------
    # DEBUG: inspect raw OmniVoice output
    # -----------------------------
    print("RAW AUDIO TYPE:", type(audio))
    if isinstance(audio, torch.Tensor):
        print("RAW AUDIO SHAPE:", audio.shape)
        print("RAW AUDIO MIN/MAX:", audio.min().item(), audio.max().item())
    elif isinstance(audio, list):
        print("RAW AUDIO LIST LENGTH:", len(audio))
        if len(audio) > 0:
            print("RAW AUDIO FIRST ELEMENT TYPE:", type(audio[0]))
    else:
        print("RAW AUDIO UNKNOWN FORMAT")

    # -----------------------------
    # Normalize OmniVoice output
    # -----------------------------
    if isinstance(audio, list):
        # Case 1: list of tensors
        if len(audio) == 1 and isinstance(audio[0], torch.Tensor):
            audio = audio[0]

        # Case 2: list of floats → convert to tensor
        elif all(isinstance(x, float) for x in audio):
            audio = torch.tensor(audio, dtype=torch.float32)

        # Case 3: list of numpy arrays → concatenate
        elif all(hasattr(x, "shape") for x in audio):
            audio = torch.tensor(np.concatenate(audio), dtype=torch.float32)

        else:
            raise TypeError(f"Unexpected audio list format: {type(audio)}")

    # Now audio MUST be a tensor
    if audio.ndim > 1:
        audio = audio.squeeze(0)

    pcm16 = (audio.clamp(-1, 1) * 32767).short().cpu().numpy()
    pcm_bytes = pcm16.tobytes()

    return base64.b64encode(pcm_bytes).decode("utf-8")
