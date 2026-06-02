import torch
import numpy as np
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

# Load VAD model once
vad_model = load_silero_vad()

def is_speech_present(audio_bytes: bytes, sample_rate: int = 16000) -> bool:
    """Return True if speech is detected in the audio chunk."""
    # bytes -> int16 -> float32 in [-1, 1]
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # NumPy -> Torch tensor (1D)
    audio_tensor = torch.from_numpy(audio_np)

    # Run VAD
    speech_ts = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=sample_rate)

    return len(speech_ts) > 0

def detect_end_of_speech(buffer: list[bytes], silence_ms: int = 800, sample_rate: int = 16000) -> bool:
    """
    Returns True if the last N milliseconds contain no speech.
    """
    if not buffer:
        return False

    # Combine last ~1 second of audio
    chunk = b"".join(buffer[-5:])  # assuming ~200ms per chunk

    # Check if speech is present
    speech = is_speech_present(chunk, sample_rate)

    return not speech
