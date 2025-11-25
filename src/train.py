# import os
# import json
# import random
# import numpy as np
# import soundfile as sf
# import librosa
# import torch
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score
# from transformers import (
#     Wav2Vec2FeatureExtractor,
#     Wav2Vec2ForSequenceClassification
# )

# # -----------------------------
# # CONFIG
# # -----------------------------
# MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
# METADATA_PATH = "data/processed/metadata.json"
# AUDIO_DIR = "data/processed/"
# SAVE_DIR = "models"
# os.makedirs(SAVE_DIR, exist_ok=True)

# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)

# # -----------------------------
# # LOAD METADATA
# # -----------------------------
# print("📄 Loading metadata...")
# with open(METADATA_PATH, "r") as f:
#     metadata = json.load(f)

# paths = [os.path.join(AUDIO_DIR, item["file"]) for item in metadata]
# labels = [item["emotion"] for item in metadata]

# # Encode labels
# label_set = sorted(list(set(labels)))
# label2id = {lbl: i for i, lbl in enumerate(label_set)}
# id2label = {i: lbl for lbl, i in label2id.items()}
# numeric_labels = [label2id[lbl] for lbl in labels]

# # -----------------------------
# # AUDIO LOADING
# # -----------------------------
# def load_audio(path):
#     audio, sr = sf.read(path)

#     # convert stereo → mono
#     if len(audio.shape) > 1:
#         audio = np.mean(audio, axis=1)

#     # resample to 16kHz
#     if sr != 16000:
#         audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

#     return audio.astype(np.float32)


# # -----------------------------
# # DATASET
# # -----------------------------
# class SERDataset(Dataset):
#     def __init__(self, paths, labels):
#         self.paths = paths
#         self.labels = labels

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, idx):
#         audio = load_audio(self.paths[idx])
#         label = self.labels[idx]
#         return audio, label


# # -----------------------------
# # FEATURE EXTRACTOR (NO TOKENIZER)
# # -----------------------------
# print("🔧 Loading Wav2Vec2 Feature Extractor...")
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)


# # -----------------------------
# # COLLATE FUNCTION
# # -----------------------------
# def collate_fn(batch):
#     audios = [item[0] for item in batch]
#     labels = torch.tensor([item[1] for item in batch])

#     inputs = feature_extractor(
#         audios,
#         sampling_rate=16000,
#         padding=True,
#         return_tensors="pt"
#     )

#     return inputs.input_values, labels


# # -----------------------------
# # LOAD MODEL
# # -----------------------------
# print("🔧 Loading Wav2Vec2 XLSR model...")
# model = Wav2Vec2ForSequenceClassification.from_pretrained(
#     MODEL_NAME,
#     num_labels=len(label_set),
#     label2id=label2id,
#     id2label=id2label,
# )

# device = "cpu"
# model = model.to(device)

# # -----------------------------
# # TRAIN/TEST SPLIT
# # -----------------------------
# train_paths, test_paths, train_labels, test_labels = train_test_split(
#     paths,
#     numeric_labels,
#     test_size=0.2,
#     random_state=SEED,
#     stratify=numeric_labels
# )

# train_ds = SERDataset(train_paths, train_labels)
# test_ds = SERDataset(test_paths, test_labels)

# train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
# test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

# # -----------------------------
# # OPTIMIZER
# # -----------------------------
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# # -----------------------------
# # TRAINING LOOP
# # -----------------------------
# EPOCHS = 3
# print("🚀 Starting Training...")

# for epoch in range(EPOCHS):
#     model.train()
#     total_loss = 0

#     for batch_inputs, batch_labels in train_loader:
#         batch_inputs = batch_inputs.to(device)
#         batch_labels = batch_labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(batch_inputs, labels=batch_labels)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"🟢 Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss/len(train_loader):.4f}")


# # -----------------------------
# # EVALUATION
# # -----------------------------
# print("\n📊 Evaluating...")
# model.eval()
# all_preds, all_true = [], []

# with torch.no_grad():
#     for batch_inputs, batch_labels in test_loader:
#         batch_inputs = batch_inputs.to(device)

#         logits = model(batch_inputs).logits
#         preds = torch.argmax(logits, dim=1)

#         all_preds.extend(preds.cpu().numpy().tolist())
#         all_true.extend(batch_labels.numpy().tolist())

# acc = accuracy_score(all_true, all_preds)
# f1 = f1_score(all_true, all_preds, average="weighted")

# print(f"\n🎯 Accuracy: {acc:.4f}")
# print(f"📈 F1-score: {f1:.4f}")

# # -----------------------------
# # SAVE MODEL
# # -----------------------------
# save_path = f"{SAVE_DIR}/wav2vec2_xlsr_crosslingual.pt"
# torch.save(model.state_dict(), save_path)

# print(f"\n💾 Model saved to: {save_path}")


import os
import json
import random
from typing import List, Dict

import numpy as np
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

# =====================================================
# CONFIG
# =====================================================
MODEL_NAME = "facebook/wav2vec2-base"       # lighter & faster than XLSR-large
METADATA_PATH = "data/processed/metadata.json"
AUDIO_DIR = "data/processed"               # where preprocess.py saved WAVs
SAVE_DIR = "models"

TARGET_SR = 16000
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-5
SEED = 42

os.makedirs(SAVE_DIR, exist_ok=True)

# =====================================================
# SEEDING
# =====================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =====================================================
# LOAD METADATA
# =====================================================
print("📄 Loading metadata...")
with open(METADATA_PATH, "r") as f:
    metadata: List[Dict] = json.load(f)

# expected metadata entry:
# {
#   "file": "ravdess_Actor_01_03-01-05-01-02-01-01.wav",
#   "emotion": "angry",
#   "language": "english",
#   "source": "RAVDESS" or "EMODB"
# }

file_names = [m["file"] for m in metadata]
emotion_labels = [m["emotion"] for m in metadata]
languages = [m.get("language", "unknown") for m in metadata]

file_paths = [os.path.join(AUDIO_DIR, fn) for fn in file_names]

# label mapping
label_set = sorted(list(set(emotion_labels)))
label2id = {lbl: i for i, lbl in enumerate(label_set)}
id2label = {i: lbl for lbl, i in label2id.items()}
numeric_labels = [label2id[lbl] for lbl in emotion_labels]

print(f"✅ Found {len(file_paths)} samples")
print(f"🎭 Emotion classes: {label_set}")
print(f"🔢 Label mapping: {label2id}")
print(f"🌍 Languages present in metadata: {sorted(list(set(languages)))}")

# =====================================================
# AUDIO LOADING (soundfile + librosa)
# =====================================================
def load_audio(path: str) -> np.ndarray:
    """Load audio as mono float32 at TARGET_SR."""
    audio, sr = sf.read(path)

    # stereo -> mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # resample if needed
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    # ensure float32
    return audio.astype(np.float32)

# =====================================================
# DATASET
# =====================================================
class SERDataset(Dataset):
    def __init__(self, paths, labels, langs):
        self.paths = paths
        self.labels = labels
        self.langs = langs

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        audio = load_audio(self.paths[idx])
        label = self.labels[idx]
        lang = self.langs[idx]
        return audio, label, lang

# =====================================================
# FEATURE EXTRACTOR
# =====================================================
print("🔧 Loading Wav2Vec2 feature extractor...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

def collate_fn(batch):
    audios = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    langs = [item[2] for item in batch]

    inputs = feature_extractor(
        audios,
        sampling_rate=TARGET_SR,
        padding=True,
        return_tensors="pt",
    )
    return inputs.input_values, labels, langs

# =====================================================
# TRAIN / TEST SPLIT (stratified by emotion)
# =====================================================
train_idx, test_idx = train_test_split(
    np.arange(len(file_paths)),
    test_size=0.2,
    random_state=SEED,
    stratify=numeric_labels,
)

train_paths = [file_paths[i] for i in train_idx]
test_paths = [file_paths[i] for i in test_idx]
train_labels = [numeric_labels[i] for i in train_idx]
test_labels = [numeric_labels[i] for i in test_idx]
train_langs = [languages[i] for i in train_idx]
test_langs = [languages[i] for i in test_idx]

print(f"📚 Train samples: {len(train_paths)}, Test samples: {len(test_paths)}")

train_ds = SERDataset(train_paths, train_labels, train_langs)
test_ds  = SERDataset(test_paths,  test_labels,  test_langs)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

# =====================================================
# CLASS WEIGHTS (handle imbalance)
# =====================================================
class_weights_np = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels,
)
class_weights = torch.tensor(class_weights_np, dtype=torch.float)
print(f"⚖ Class weights: {class_weights_np}")

# =====================================================
# MODEL
# =====================================================
print("🔧 Loading Wav2Vec2 base model...")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_set),
    label2id=label2id,
    id2label=id2label,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
class_weights = class_weights.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# =====================================================
# TRAINING LOOP
# =====================================================
print("🚀 Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch_inputs, batch_labels, _batch_langs in train_loader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_inputs, labels=None)
        logits = outputs.logits
        loss = loss_fn(logits, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"🟢 Epoch {epoch + 1}/{EPOCHS} — Loss: {avg_loss:.4f}")

# =====================================================
# EVALUATION HELPERS
# =====================================================
def evaluate_model(loader, desc: str):
    model.eval()
    all_preds = []
    all_true = []
    all_langs = []

    with torch.no_grad():
        for batch_inputs, batch_labels, batch_langs in loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_true.extend(batch_labels.cpu().numpy().tolist())
            all_langs.extend(batch_langs)

    acc = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds, average="weighted")
    print(f"\n📊 {desc}")
    print(f"🎯 Accuracy: {acc:.4f}")
    print(f"📈 Weighted F1-score: {f1:.4f}")
    print("📌 Detailed report:")
    print(classification_report(all_true, all_preds, target_names=[id2label[i] for i in range(len(label_set))]))
    return np.array(all_true), np.array(all_preds), np.array(all_langs)

def evaluate_by_language(true, preds, langs, target_lang: str):
    mask = langs == target_lang
    if mask.sum() == 0:
        print(f"\n⚠ No samples for language '{target_lang}' in this split.")
        return

    t = true[mask]
    p = preds[mask]
    acc = accuracy_score(t, p)
    f1 = f1_score(t, p, average="weighted")
    print(f"\n🌐 Language-specific metrics for {target_lang}:")
    print(f"🎯 Accuracy: {acc:.4f}")
    print(f"📈 Weighted F1-score: {f1:.4f}")
    print(classification_report(t, p, target_names=[id2label[i] for i in range(len(label_set))]))

# =====================================================
# EVALUATION
# =====================================================
y_true, y_pred, y_langs = evaluate_model(test_loader, "Overall test-set performance")

# Language-wise breakdown (works once EMO-DB is added)
for lang_name in sorted(set(y_langs)):
    evaluate_by_language(y_true, y_pred, y_langs, lang_name)

# =====================================================
# SAVE MODEL
# =====================================================
save_path = os.path.join(SAVE_DIR, "wav2vec2_base_crosslingual_ser.pt")
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "label2id": label2id,
        "id2label": id2label,
        "model_name": MODEL_NAME,
    },
    save_path,
)
print(f"\n💾 Model saved to: {save_path}")
