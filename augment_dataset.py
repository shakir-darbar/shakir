"""
augment_dataset.py -- Offline Data Augmentation (SpecAugment)

Applies class-specific multipliers to the training dataset to resolve
class imbalance. Generates new synthetic samples using SpecAugment 
(Time and Frequency masking) on the Mel-spectrogram PNGs.

Validation and Test sets are copied EXACTLY as-is to prevent data leakage.
"""

import os
import shutil
import random
import numpy as np
from PIL import Image, ImageEnhance

SRC_DIR = os.path.join('data', 'original_png')
DST_DIR = os.path.join('data', 'augmented_png')

TARGET_CLASSES = ['Healthy', 'Pneumonia', 'URTI', 'Bronchiectasis']

# Multipliers: (1 means just keep the original, 2 means 1 original + 1 augmented, etc.)
CLASS_MULTIPLIERS = {
    'Healthy': 2,
    'Pneumonia': 2,
    'URTI': 3,
    'Bronchiectasis': 10
}

def apply_spec_augment(image, max_time_mask=30, max_freq_mask=30):
    """Applies SpecAugment-style masking to a PIL image."""
    img_arr = np.array(image)
    h, w, c = img_arr.shape
    
    aug_arr = img_arr.copy()
    
    # 1. Frequency Masking (Horizontal blocks)
    if random.random() > 0.3:
        f = random.randint(5, max_freq_mask)
        f0 = random.randint(0, h - f)
        aug_arr[f0:f0+f, :, :] = 0  # Blackout

    # 2. Time Masking (Vertical blocks)
    if random.random() > 0.3:
        t = random.randint(5, max_time_mask)
        t0 = random.randint(0, w - t)
        aug_arr[:, t0:t0+t, :] = 0  # Blackout
        
    aug_img = Image.fromarray(aug_arr)
    
    # 3. Brightness/Contrast Jitter
    if random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(aug_img)
        aug_img = enhancer.enhance(random.uniform(0.7, 1.3))
    
    return aug_img


def process_train_set():
    src_train = os.path.join(SRC_DIR, 'train')
    dst_train = os.path.join(DST_DIR, 'train')
    os.makedirs(dst_train, exist_ok=True)
    
    print("\n[1/3] Augmenting Training Set...")
    
    for cls in TARGET_CLASSES:
        src_cls_dir = os.path.join(src_train, cls)
        dst_cls_dir = os.path.join(dst_train, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)
        
        if not os.path.exists(src_cls_dir):
            continue
            
        files = [f for f in os.listdir(src_cls_dir) if f.lower().endswith('.png')]
        multiplier = CLASS_MULTIPLIERS.get(cls, 1)
        
        print(f"  -> {cls:<15}: {len(files)} originals | Multiplier: x{multiplier} | Target: ~{len(files)*multiplier}")
        
        for file in files:
            src_path = os.path.join(src_cls_dir, file)
            
            # 1. Copy the original unaltered
            dst_path_orig = os.path.join(dst_cls_dir, file)
            shutil.copy2(src_path, dst_path_orig)
            
            # 2. Generate (multiplier - 1) augmented copies
            try:
                with Image.open(src_path) as img:
                    img = img.convert('RGB')
                    for i in range(1, multiplier):
                        aug_img = apply_spec_augment(img)
                        base, ext = os.path.splitext(file)
                        aug_filename = f"{base}_aug_{i}{ext}"
                        aug_path = os.path.join(dst_cls_dir, aug_filename)
                        aug_img.save(aug_path)
            except Exception as e:
                print(f"     Error processing {file}: {e}")

def copy_unaltered_set(split_name):
    print(f"[{'2' if split_name=='validation' else '3'}/3] Copying {split_name.capitalize()} Set UNALTERED...")
    src_split = os.path.join(SRC_DIR, split_name)
    dst_split = os.path.join(DST_DIR, split_name)
    
    if os.path.exists(dst_split):
        shutil.rmtree(dst_split)
        
    shutil.copytree(src_split, dst_split)
    print(f"  -> Successfully copied {split_name} (NO augmentation applied to prevent leakage).")


def main():
    print("=" * 60)
    print("  PHASE 2: OFFLINE DATA AUGMENTATION (SpecAugment)")
    print("=" * 60)
    
    if not os.path.exists(SRC_DIR):
        print(f"Error: Source directory {SRC_DIR} not found.")
        return
        
    process_train_set()
    copy_unaltered_set('validation')
    copy_unaltered_set('test')
    
    print("\n=" * 60)
    print("  AUGMENTATION COMPLETE")
    print(f"  New Dataset Location: {DST_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
