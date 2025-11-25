import json
import torch
import soundfile as sf
from sklearn.metrics import accuracy_score, f1_score
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

MODEL_PATH = "models/wav2vec2_base_crosslingual_ser.pt"
METADATA_PATH = "data/processed/metadata.json"
AUDIO_DIR = "data/processed/audio"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=4
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Load feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# Label mapping
label2id = {"angry": 0, "happy": 1, "neutral": 2, "sad": 3}
id2label = {v: k for k, v in label2id.items()}

# Load metadata
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

def load_audio(path):
    audio, sr = sf.read(path)
    return audio

def evaluate_language(lang):
    preds, labels = [], []

    for item in metadata:
        if item["language"] != lang:
            continue

        path = f"{AUDIO_DIR}/{item['file']}"
        audio = load_audio(path)

        inputs = feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits

        pred = torch.argmax(logits, dim=-1).cpu().item()
        preds.append(pred)
        labels.append(label2id[item["emotion"]])

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')

    return acc, f1, len(labels)

# Run evaluations
eng_acc, eng_f1, n_eng = evaluate_language("english")
ger_acc, ger_f1, n_ger = evaluate_language("german")

print("\n========== LANGUAGE-WISE PERFORMANCE ==========\n")
print(f"🇬🇧 English-only  ({n_eng} samples)")
print(f"Accuracy: {eng_acc:.4f}")
print(f"F1-score: {eng_f1:.4f}\n")

print(f"🇩🇪 German-only   ({n_ger} samples)")
print(f"Accuracy: {ger_acc:.4f}")
print(f"F1-score: {ger_f1:.4f}\n")
