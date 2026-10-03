"""
visualize_spectrograms.py -- Visualize Mel-Spectrogram Examples for Target Classes

Loads one sample PNG spectrogram for each of the 4 target classes:
    1. Healthy
    2. URTI (Upper Respiratory Tract Infection)
    3. Pneumonia
    4. Bronchiectasis

Generates a clean comparison figure and saves it to plots/spectrogram_examples.png.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

TARGET_CLASSES = ['Healthy', 'URTI', 'Pneumonia', 'Bronchiectasis']
DATA_DIR = os.path.join('data', 'original_png', 'train')
OUTPUT_DIR = 'plots'


def visualize_target_spectrograms():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, cls in enumerate(TARGET_CLASSES):
        cls_dir = os.path.join(DATA_DIR, cls)
        if not os.path.exists(cls_dir):
            print(f"Warning: Directory not found: {cls_dir}")
            continue

        png_files = [f for f in os.listdir(cls_dir) if f.lower().endswith('.png')]
        if not png_files:
            print(f"Warning: No PNGs found in {cls_dir}")
            continue

        sample_path = os.path.join(cls_dir, png_files[0])
        img = Image.open(sample_path)

        axes[idx].imshow(img)
        axes[idx].set_title(f"Class: {cls}\nSample: {png_files[0]}", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel("Time Frames (~157 frames / 5 sec)", fontsize=10)
        axes[idx].set_ylabel("Mel Frequency Bins (128)", fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'spectrogram_examples.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[OK] Spectrogram examples visualization saved to: {output_path}")
    return output_path


if __name__ == '__main__':
    visualize_target_spectrograms()
