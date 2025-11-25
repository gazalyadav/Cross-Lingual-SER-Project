🎙️ Cross-Lingual Speech Emotion Recognition (SER) System
🚀 Overview

This project is an AI-powered Speech Emotion Recognition (SER) system capable of identifying human emotions such as Angry, Happy, Neutral, and Sad from raw audio.
It supports cross-lingual emotion classification using two benchmark datasets:
RAVDESS (English)
Emo-DB (German)

The project includes a full pipeline from preprocessing → training → evaluation → real-time inference UI (Gradio).
Designed for emotion-aware AI, call centers, healthcare monitoring, virtual assistants, and mental health analysis.

🎯 Features:
✅ Cross-Lingual Emotion Recognition – Works on English + German
🎤 Raw Audio Input – No MFCCs required
🧠 Transformer-Based Model – Wav2Vec2 (Facebook AI)
⚡ High Accuracy (~90%) – On combined test set
📚 Automatic Metadata Generation
🧹 Advanced Preprocessing – Resampling, trimming, normalization
📊 Test Results – Includes per-language performance
🎛️ Real-Time Emotion Detector – Microphone + File input
🌐 Gradio-Based UI for deployment

🏗️ Tech Stack:
🔹 Python – Core programming
🔹 PyTorch – Deep learning framework
🔹 HuggingFace Transformers – Wav2Vec2 model
🔹 Librosa / SoundFile – Audio loading + processing
🔹 Scikit-learn – Metrics + train/test split
🔹 Gradio – Real-time inference interface

📦 Installation

Clone the repository and install dependencies:

git clone https://github.com/gazalyadav/Cross-Lingual-SER-Project.git
cd Cross-Lingual-SER-Project
pip install -r requirements.txt

▶️ Running the Project
1️⃣ Preprocess the datasets

This step loads RAVDESS + Emo-DB, resamples audio to 16kHz, normalizes it, and creates metadata.

python src/preprocess.py

Output is stored in:

data/processed/
    ├── *.wav
    └── metadata.json

2️⃣ Train the Wav2Vec2 SER Model
python src/train.py

Expected Results
Dataset	Accuracy	Weighted F1
English	~91%	~0.91
German	~87%	~0.87
Combined	~89–90%	~0.90

3️⃣ Run the Real-Time Gradio App
python src/app_gradio.py


App starts at:
🔗 http://127.0.0.1:7860

You can:
🎤 Speak using Microphone
📁 Upload a .wav file
📊 View predicted emotion instantly


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

🏆 How It Works
1. Start Preprocessing
Loads RAVDESS + Emo-DB → Converts to mono → Resamples to 16 kHz → Normalizes → Saves processed files.

2. Train the SER Model
Wav2Vec2 extracts features directly from raw waveforms → Softmax classifier predicts emotions.

3. Test the Model
Calculates Accuracy + F1 + per-language performance (English/German).

4. Real-Time Prediction
You speak → Audio processed → Wav2Vec2 inference → Emotion displayed instantly.

🚀 Future Enhancements

📌 Add Hindi + Multilingual Indian datasets
📌 Add gender + speaker ID
📌 Deploy on HuggingFace Spaces
📌 Convert to ONNX for mobile deployment
📌 Add live streaming via WebSockets

🤝 Contributing
Contributions are welcome!
Fork → Create a branch → Commit → Open PR.

🔗 License
MIT License – Free to use and modify.

🎓 Author
Gazall Yadav
AI/ML Developer | SER Researcher
🔗 GitHub: https://github.com/gazalyadav




