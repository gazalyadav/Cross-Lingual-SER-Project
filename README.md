🎙️ Cross-Lingual Speech Emotion Recognition (SER) Using Wav2Vec2
🚀 Overview

This project is an AI-powered emotion recognition system that analyzes human speech and predicts emotional states such as Angry, Happy, Neutral, and Sad.
It is built using state-of-the-art Transformer-based models (Wav2Vec2) and supports cross-lingual emotion recognition across English (RAVDESS) and German (Emo-DB) speech.

The system includes a complete pipeline:
📥 Data preprocessing → 🧠 Model training → 🎧 Real-time inference → 🌐 Deployment.

It is designed for applications such as call centers, healthcare monitoring, virtual assistants, mental health analysis, and emotionally aware AI systems.

🎯 Features

✅ Cross-Lingual Emotion Recognition – Works across English + German datasets
🎤 Raw Audio Input – No MFCCs required
🧠 Transformer-based Model – Uses Wav2Vec2 (Facebook AI)
⚡ High Accuracy – ~90% accuracy on combined test set
📈 Balanced Label Mapping – Unified emotion labels across datasets
🔊 Real-Time Emotion Detection App – Built using Gradio
📂 Metadata & Processed Audio Generation
🛠 Robust Preprocessing Pipeline – Resampling, trimming, normalization

🏗️ Tech Stack

🔹 Python
🔹 PyTorch
🔹 HuggingFace Transformers
🔹 Torchaudio / Librosa
🔹 Scikit-learn
🔹 Gradio (Real-time inference UI)

📦 Installation
Clone the repository
git clone https://github.com/gazalyadav/Cross-Lingual-SER-Project.git
cd Cross-Lingual-SER-Project

Create Conda environment
conda create -n ser python=3.10
conda activate ser

Install dependencies
pip install -r requirements.txt

▶️ Running the Project
1️⃣ Preprocess the datasets

This step loads RAVDESS + Emo-DB, resamples audio, normalizes, and generates metadata.

python src/preprocess.py


It creates:

data/processed/
     ├── *.wav
     └── metadata.json

2️⃣ Train the SER Model
python src/train.py


Expected output:

Dataset	Accuracy	Weighted F1
English	~91%	~0.91
German	~87%	~0.87
Combined	~89–90%	~0.90
3️⃣ Run the Real-Time App
python src/app_gradio.py


You can now use:

🎤 Microphone recording
🔊 WAV file upload
📊 Instant emotion prediction

Runs locally at:

http://127.0.0.1:7860

📸 Screenshots (Add Yours)

You may add screenshots like this:

or upload snapshots of your Gradio UI / terminal output / project structure.

📑 File Structure
CrossLingual_SER/
│
├── data/
│   ├── raw/
│   │   ├── RAVDESS/
│   │   └── EMODB/
│   └── processed/
│       ├── *.wav
│       └── metadata.json
│
├── models/
│   └── wav2vec2_base_crosslingual_ser.pt
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── app_gradio.py
│   ├── utils.py
│   └── …
│
├── requirements.txt
└── README.md

🧠 How It Works (Pipeline)

Data Input
Loads emotional speech from RAVDESS & Emo-DB.

Preprocessing
✔ Resampling to 16 kHz
✔ Mono conversion
✔ Trimming
✔ Normalization
✔ Label harmonization

Feature Extraction
Transformer extracts contextual features from raw waveforms.

Training

Wav2Vec2-base model

AdamW optimizer

Balanced class weights

10 epochs

Prediction
Real-time microphone or audio upload → Wav2Vec2 → Emotion output.

Deployment
Gradio app for instant demo.

🚀 Future Enhancements

📌 Hindi + Multi-Indian-language Dataset Support
📌 Add Gender Detection + Emotion Fusion
📌 Convert model to ONNX for mobile apps
📌 Deploy on HuggingFace Spaces
📌 Add real-time streaming (WebSocket)

🤝 Contributing

Contributions are welcome!
You can fork the project, create a branch, and submit a pull request.

🔗 License

MIT License — free to use and modify.

🎓 Author

Gazall Yadav
AI/ML Developer | SER Researcher | Emotion-Aware Systems
GitHub: https://github.com/gazalyadav
