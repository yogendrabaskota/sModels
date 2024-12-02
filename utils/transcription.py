from transformers import AutoProcessor, AutoModelForCTC
import torch
import soundfile as sf
import os

# Load the model and processor
MODEL_NAME = "iamTangsang/Wav2Vec2_XLS-R-300m_Nepali_ASR"
MODEL_DIR = "./models"

processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)
model = AutoModelForCTC.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)

def transcribe_audio(audio_path):
    # Load the audio file
    audio, sampling_rate = sf.read(audio_path)

    # Preprocess the audio
    inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    # Forward pass through the model
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    # Decode predictions
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription[0]
