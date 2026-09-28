import sys
sys.path.insert(0, '.')
import os
os.environ['KERAS_BACKEND'] = 'torch'
import keras
import numpy as np
import glob
from src.preprocessing import load_and_preprocess_audio, segment_audio, extract_mel_spectrogram
from src.data_loader import parse_icbhi_filename, load_diagnosis_mapping

import pandas as pd
df_diag = pd.read_csv('data/raw/diagnosis.csv', header=None, names=['patient_id', 'diagnosis'])
mapping = dict(zip(df_diag['patient_id'], df_diag['diagnosis']))
model = keras.models.load_model('best_model.h5')

files = glob.glob('data/raw/*.wav')[:20]
classes = ['Healthy', 'Pneumonia', 'URTI', 'Bronchiectasis']

print(f"{'Filename':<30} | {'True Diagnosis':<15} | {'Predicted':<15} | Probabilities")
print("-" * 90)

for f in files:
    pid = parse_icbhi_filename(f)
    true_diag = mapping.get(pid, 'Unknown')
    audio = load_and_preprocess_audio(f)
    segs = segment_audio(audio)
    specs = np.expand_dims(np.array([extract_mel_spectrogram(s) for s in segs]), -1)
    
    # Check shape
    if specs.shape[2] != 157:
        # Resize or pad/crop to 157
        pass

    segment_probs = model.predict(specs, verbose=0)
    probs = np.mean(segment_probs, axis=0)
    pred = classes[np.argmax(probs)]
    print(f"{os.path.basename(f):<30} | {true_diag:<15} | {pred:<15} | {np.round(probs, 3)}")
