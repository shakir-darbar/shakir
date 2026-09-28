import sys
sys.path.insert(0, '.')
import os
os.environ['KERAS_BACKEND'] = 'torch'
import keras
import numpy as np
import pandas as pd

from src.preprocessing import load_and_preprocess_audio, segment_audio, extract_mel_spectrogram
from src.data_loader import build_dataset_manifest, TARGET_CLASSES, CLASS_TO_IDX

df_manifest = build_dataset_manifest('data/raw')
print(f"Total manifest samples matching 4 target classes: {len(df_manifest)}")

model_mobilenet = keras.models.load_model('best_model.h5')
model_cnn = keras.models.load_model('models/checkpoints/custom_cnn_best.h5')

mobilenet_correct = 0
cnn_correct = 0
total = 0

print(f"\n{'Filename':<35} | {'True Label':<15} | {'MobileNet Pred':<15} | {'CNN Pred':<15}")
print("-" * 90)

for idx, row in df_manifest.iterrows():
    filepath = row['filepath']
    true_label = row['label']
    true_idx = row['label_idx']
    
    audio = load_and_preprocess_audio(filepath)
    segments = segment_audio(audio)
    if len(segments) == 0:
        continue
        
    specs = np.expand_dims(np.array([extract_mel_spectrogram(s) for s in segments]), -1)
    
    # Predict MobileNet
    m_probs = np.mean(model_mobilenet.predict(specs, verbose=0), axis=0)
    m_pred_idx = np.argmax(m_probs)
    m_pred_label = TARGET_CLASSES[m_pred_idx]
    
    # Predict CNN
    c_probs = np.mean(model_cnn.predict(specs, verbose=0), axis=0)
    c_pred_idx = np.argmax(c_probs)
    c_pred_label = TARGET_CLASSES[c_pred_idx]
    
    if m_pred_idx == true_idx:
        mobilenet_correct += 1
    if c_pred_idx == true_idx:
        cnn_correct += 1
    total += 1
    
    print(f"{os.path.basename(filepath):<35} | {true_label:<15} | {m_pred_label:<15} | {c_pred_label:<15}")

print("\n" + "="*60)
print(f"Total Target Class Files Evaluated: {total}")
print(f"MobileNetV2 Accuracy: {mobilenet_correct}/{total} ({mobilenet_correct/total*100:.2f}%)")
print(f"Custom CNN Accuracy : {cnn_correct}/{total} ({cnn_correct/total*100:.2f}%)")
print("="*60)
