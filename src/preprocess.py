import os
import json
import librosa
import soundfile as sf
from tqdm import tqdm

# ----------------------------
# PATHS
# ----------------------------
RAW_RAVDESS = "data/raw/RAVDESS"
RAW_EMODB = "data/raw/EMODB/wav"
PROCESSED = "data/processed"
META_PATH = "data/processed/metadata.json"

TARGET_SR = 16000  # standard sample rate

# ----------------------------
# RAVDESS EMOTION MAP
# ----------------------------
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# ----------------------------
# EMO-DB EMOTION MAP
# ----------------------------
# Example filename: 03a01Fa.wav
# The **5th character** defines emotion:
# F = anger, W = disgust, L = boredom, T = fear, N = neutral, E = joy, A = sadness
EMODB_EMOTION_MAP = {
    "W": "disgust",
    "L": "boredom",
    "E": "happy",
    "A": "sad",
    "F": "angry",
    "T": "fearful",
    "N": "neutral"
}

VALID_EMOTIONS = ["neutral", "happy", "sad", "angry"]

# ----------------------------
# UTILITY
# ----------------------------
def save_audio(audio, sr, filename):
    os.makedirs(PROCESSED, exist_ok=True)
    path = os.path.join(PROCESSED, filename)
    sf.write(path, audio, sr)
    return path

# ----------------------------
# RAVDESS PARSER
# ----------------------------
def parse_ravdess_emotion(filename):
    parts = filename.split("-")
    code = parts[2]
    emotion = RAVDESS_EMOTION_MAP.get(code)
    if emotion not in VALID_EMOTIONS:
        return None
    return emotion

# ----------------------------
# EMO-DB PARSER
# ----------------------------
def parse_emodb_emotion(filename):
    """
    EMO-DB naming: 03a01Fa.wav → 5th letter = F (emotion)
    """
    if len(filename) < 6:
        return None

    code = filename[5].upper()
    emotion = EMODB_EMOTION_MAP.get(code)

    if emotion not in VALID_EMOTIONS:
        return None
    return emotion

# ----------------------------
# PROCESS RAVDESS
# ----------------------------
def preprocess_ravdess(metadata):
    if not os.path.exists(RAW_RAVDESS):
        print("⚠ RAVDESS folder not found, skipping...")
        return metadata

    print("\n🔄 Processing RAVDESS audio...\n")

    for actor in sorted(os.listdir(RAW_RAVDESS)):
        actor_dir = os.path.join(RAW_RAVDESS, actor)
        if not os.path.isdir(actor_dir):
            continue

        for file in tqdm(os.listdir(actor_dir), desc=f"RAVDESS {actor}"):
            if not file.endswith(".wav"):
                continue

            emotion = parse_ravdess_emotion(file)
            if emotion is None:
                continue

            path = os.path.join(actor_dir, file)

            audio, sr = librosa.load(path, sr=TARGET_SR)
            audio = audio / max(abs(audio))

            new_name = f"ravdess_{actor}_{file}"
            save_audio(audio, TARGET_SR, new_name)

            metadata.append({
                "file": new_name,
                "emotion": emotion,
                "language": "english",
                "source": "RAVDESS"
            })

    return metadata

# ----------------------------
# PROCESS EMO-DB
# ----------------------------
def preprocess_emodb(metadata):
    if not os.path.exists(RAW_EMODB):
        print("⚠ EMO-DB folder not found, skipping...")
        return metadata

    print("\n🔄 Processing EMO-DB audio...\n")

    for file in tqdm(os.listdir(RAW_EMODB), desc="EMO-DB"):
        if not file.endswith(".wav"):
            continue

        emotion = parse_emodb_emotion(file)
        if emotion is None:
            continue

        path = os.path.join(RAW_EMODB, file)

        audio, sr = librosa.load(path, sr=TARGET_SR)
        audio = audio / max(abs(audio))

        new_name = f"emodb_{file}"
        save_audio(audio, TARGET_SR, new_name)

        metadata.append({
            "file": new_name,
            "emotion": emotion,
            "language": "german",
            "source": "EMO-DB"
        })

    return metadata

# ----------------------------
# MAIN PIPELINE
# ----------------------------
if __name__ == "__main__":
    os.makedirs(PROCESSED, exist_ok=True)

    metadata = []

    metadata = preprocess_ravdess(metadata)
    metadata = preprocess_emodb(metadata)

    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n✅ Preprocessing Complete!")
    print(f"📁 Processed audio saved in: {PROCESSED}")
    print(f"📄 Metadata saved at: {META_PATH}")
