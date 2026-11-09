import os
import pickle
import warnings
import librosa
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "siren_classifier.pkl")
DEFAULT_AUDIO_PATH = os.path.join(BASE_DIR, "data", "audio", "non-siren", "sound_401.wav")


def extract_mfcc(audio_path, duration=3):
    y, sr = librosa.load(audio_path, duration=duration)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)


def predict_siren(audio_path=DEFAULT_AUDIO_PATH):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Classifier not found at: {MODEL_PATH}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(MODEL_PATH, "rb") as f:
            clf = pickle.load(f)

    print(f"Extracting MFCC features from: {audio_path}")
    features = extract_mfcc(audio_path)
    prediction = clf.predict([features])[0]

    if prediction == 1:
        print("[RESULT] SIREN DETECTED! (Emergency Vehicle Siren)")
    else:
        print("[RESULT] No siren detected (Standard traffic / background)")
    return prediction


if __name__ == "__main__":
    predict_siren()
