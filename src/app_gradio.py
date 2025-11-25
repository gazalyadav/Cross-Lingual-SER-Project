import os
import torch
import numpy as np
import soundfile as sf
import librosa
import gradio as gr
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

MODEL_PATH = "models/wav2vec2_base_crosslingual_ser.pt"

print("🔧 Loading model...")
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

label2id = checkpoint["label2id"]
id2label = checkpoint["id2label"]
model_name = checkpoint["model_name"]

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label,
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def load_audio(filepath):
    audio, sr = sf.read(filepath)

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    return audio.astype(np.float32)


def predict(audio_tuple):
    try:
        if audio_tuple is None:
            return "No audio detected", {}

        filepath = audio_tuple  # Gradio returns filepath in non-streaming mode
        audio = load_audio(filepath)

        inputs = feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            logits = model(inputs.input_values).logits
            probs = torch.softmax(logits, dim=-1)[0].numpy()

        pred = id2label[int(np.argmax(probs))]
        prob_dict = {id2label[i]: float(probs[i]) for i in range(len(probs))}

        return pred, prob_dict

    except Exception as e:
        return f"Error: {str(e)}", {}


with gr.Blocks() as app:
    gr.Markdown("## 🎙️ Cross-Lingual Speech Emotion Recognition")

    audio_input = gr.Audio(
        sources=["microphone", "upload"],
        type="filepath",
        streaming=False,
        label="Speak or upload audio"
    )

    pred_output = gr.Textbox(label="Predicted Emotion")
    prob_output = gr.Label(label="Emotion Probabilities")

    audio_input.change(
        fn=predict,
        inputs=audio_input,
        outputs=[pred_output, prob_output]
    )

app.launch()
