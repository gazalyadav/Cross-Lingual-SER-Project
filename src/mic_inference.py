import sounddevice as sd

def record_from_mic(duration_sec=3, sr=TARGET_SR):
    """Record audio from microphone for duration_sec seconds."""
    print(f"🎙 Recording {duration_sec} seconds... Speak now!")
    audio = sd.rec(
        int(duration_sec * sr),
        samplerate=sr,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    audio = audio.squeeze()
    return audio


def predict_from_mic(duration_sec=3):
    model, feature_extractor, id2label, device = load_model()

    audio = record_from_mic(duration_sec)
    inputs = feature_extractor(
        [audio],
        sampling_rate=TARGET_SR,
        padding=True,
        return_tensors="pt",
    )
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    pred_label = id2label[pred_id]
    prob_dict = {id2label[i]: float(p) for i, p in enumerate(probs)}

    print_prediction(pred_label, prob_dict)
