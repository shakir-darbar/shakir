"""
generate_spectrogram_images.py
Generates presentation-ready spectrogram images for all 4 disease classes
and a comprehensive 4-class acoustic comparison figure.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from src.preprocessing import load_and_preprocess_audio, segment_audio, extract_mel_spectrogram, SAMPLE_RATE

OUTPUT_DIR = os.path.join("plots", "spectrograms")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = ["Healthy", "Pneumonia", "URTI", "Bronchiectasis"]
sample_files = {}

for cls in CLASSES:
    class_folder = os.path.join("data", "test_wavs", cls)
    wavs = glob.glob(os.path.join(class_folder, "*.wav"))
    if wavs:
        sample_files[cls] = wavs[0]
    else:
        # Fallback to search in data/raw
        pass

print("Selected representative samples:")
for cls, path in sample_files.items():
    print(f"  {cls}: {os.path.basename(path)}")

# 1. Generate Individual Detailed 3-Channel Breakdown for each class
for cls, filepath in sample_files.items():
    audio = load_and_preprocess_audio(filepath)
    segments = segment_audio(audio)
    seg = segments[0] # Take first 5-second segment
    
    # Compute Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=seg, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    delta = librosa.feature.delta(log_mel, order=1)
    delta2 = librosa.feature.delta(log_mel, order=2)
    
    # Plot detailed 3-channel breakdown
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=False)
    fig.suptitle(f"Acoustic Feature Extraction Pipeline: {cls}", fontsize=15, fontweight='bold', y=0.98)
    
    # Row 1: Raw Waveform
    time_axis = np.linspace(0, len(seg)/SAMPLE_RATE, len(seg))
    axes[0].plot(time_axis, seg, color='#2b5c8f', lw=1)
    axes[0].set_title(f"1. Raw Audio Waveform (5-Second Window, 16 kHz Mono)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, 5)
    axes[0].grid(True, alpha=0.3)
    
    # Row 2: Channel 0 - Log-Mel Spectrogram (Static)
    img1 = librosa.display.specshow(log_mel, sr=SAMPLE_RATE, hop_length=512, x_axis='time', y_axis='mel', ax=axes[1], cmap='magma')
    axes[1].set_title(f"2. Channel 0: Log-Mel Spectrogram (Static Acoustic Energy in dB)", fontsize=11, fontweight='bold')
    fig.colorbar(img1, ax=axes[1], format="%+2.0f dB")
    
    # Row 3: Channel 1 - Delta (Velocity)
    img2 = librosa.display.specshow(delta, sr=SAMPLE_RATE, hop_length=512, x_axis='time', y_axis='mel', ax=axes[2], cmap='coolwarm')
    axes[2].set_title(f"3. Channel 1: Delta Spectrogram (1st Derivative / Frequency Velocity)", fontsize=11, fontweight='bold')
    fig.colorbar(img2, ax=axes[2])
    
    # Row 4: Channel 2 - Delta-Delta (Acceleration)
    img3 = librosa.display.specshow(delta2, sr=SAMPLE_RATE, hop_length=512, x_axis='time', y_axis='mel', ax=axes[3], cmap='viridis')
    axes[3].set_title(f"4. Channel 2: Delta-Delta Spectrogram (2nd Derivative / Acceleration)", fontsize=11, fontweight='bold')
    axes[3].set_xlabel("Time (seconds)", fontsize=11)
    fig.colorbar(img3, ax=axes[3])
    
    plt.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, f"feature_breakdown_{cls.lower()}.png")
    plt.savefig(out_file, dpi=250)
    plt.close()
    print(f"Saved: {out_file}")

# 2. Master Comparison Figure (All 4 classes side-by-side)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

class_descriptions = {
    "Healthy": "Uniform, smooth harmonic energy; absence of high-frequency adventitious peaks.",
    "Pneumonia": "Coarse crackle bursts (explosive vertical energy streaks across mid/high frequencies).",
    "URTI": "High-frequency upper airway turbulence & periodic cough spectral spikes.",
    "Bronchiectasis": "Distinct biphasic wheezes & crackles (intense localized horizontal & vertical bands)."
}

for idx, cls in enumerate(CLASSES):
    filepath = sample_files[cls]
    audio = load_and_preprocess_audio(filepath)
    segments = segment_audio(audio)
    seg = segments[0]
    
    mel_spec = librosa.feature.melspectrogram(y=seg, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    ax = axes[idx]
    img = librosa.display.specshow(log_mel, sr=SAMPLE_RATE, hop_length=512, x_axis='time', y_axis='mel', ax=ax, cmap='inferno')
    ax.set_title(f"Class: {cls}\n({class_descriptions[cls]})", fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Frequency (Hz)", fontsize=10)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

plt.suptitle("Comparative Log-Mel Spectrogram Signatures Across 4 Respiratory Classes", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
comparison_path = os.path.join(OUTPUT_DIR, "all_4_diseases_spectrogram_comparison.png")
plt.savefig(comparison_path, dpi=300)
plt.close()
print(f"\nSaved Master Comparison Figure: {comparison_path}")
