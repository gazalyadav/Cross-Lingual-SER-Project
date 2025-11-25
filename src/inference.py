import os
import argparse
import json
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

# -----------------------------
# CONFIG
# -----------------------------
CHECKPOINT_PATH = "models/wav2vec2_base_crosslingual_ser.pt"
TARGET_SR = 16000

# -----------------------------
# AUDIO LOADING (same as train.py)
# -----------------------------
def load_audio(path, target_sr=TARGET_SR):
    """Load an audio file, convert to mono and resample to target_sr."""
    audio, sr = sf.read(path)

    # stereo -> mono
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio.astype(np.float32)


# -----------------------------
# MODEL LOADING
# -----------------------------
def load_model(device=None):
    """Load Wav2Vec2 model + feature extractor + label maps from checkpoint."""
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    print(f"🔧 Loading checkpoint from {CHECKPOINT_PATH} ...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    model_name = checkpoint.get("model_name", "facebook/wav2vec2-base")
    label2id = checkpoint["label2id"]
    id2label = {int(k): v for k, v in checkpoint["id2label"].items()}

    print(f"📦 Base model: {model_name}")
    print(f"🎭 Emotion labels: {id2label}")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, feature_extractor, id2label, device


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_emotion(audio_path, model, feature_extractor, id2label, device):
    """Run a single prediction on one audio file."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load and preprocess audio
    audio = load_audio(audio_path)
    inputs = feature_extractor(
        [audio],
        sampling_rate=TARGET_SR,
        padding=True,
        return_tensors="pt",
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    pred_label = id2label[pred_id]

    # make a {label: prob} dictionary
    prob_dict = {id2label[i]: float(p) for i, p in enumerate(probs)}

    return pred_label, prob_dict


# -----------------------------
# OPTIONAL: PRETTY PRINT
# -----------------------------
def print_prediction(pred_label, prob_dict):
    print(f"\n🔊 Predicted emotion: **{pred_label.upper()}**\n")
    print("📊 Probabilities:")
    for lbl, p in prob_dict.items():
        print(f"  - {lbl:8s}: {p:.3f}")


# -----------------------------
# CLI ENTRY POINT
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Cross-lingual SER Inference")
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to .wav file",
    )
    args = parser.parse_args()

    # Load model once
    model, feature_extractor, id2label, device = load_model()

    # Predict
    pred_label, prob_dict = predict_emotion(
        args.audio, model, feature_extractor, id2label, device
    )

    print_prediction(pred_label, prob_dict)


if __name__ == "__main__":
    main()
