"""
data_pipeline.py -- PNG Image Dataset Loading Pipeline

Loads Mel-spectrogram PNG images from the organized directory structure
and prepares TensorFlow datasets for training, validation, and testing.

Directory structure expected:
    data/original_png/
        train/       {Healthy, URTI, Pneumonia, Bronchiectasis}/
        validation/  {Healthy, URTI, Pneumonia, Bronchiectasis}/
        test/        {Healthy, URTI, Pneumonia, Bronchiectasis}/
"""

import os
import numpy as np
import tensorflow as tf
from collections import Counter


# Target classes in the fixed order matching label_mapping.json
TARGET_CLASSES = ['Healthy', 'Pneumonia', 'URTI', 'Bronchiectasis']
NUM_CLASSES = len(TARGET_CLASSES)


def load_dataset(data_dir, image_size=(224, 224), batch_size=32, split='train', seed=42):
    """
    Loads PNG images from a split directory using tf.keras.utils.image_dataset_from_directory.

    Parameters:
        data_dir (str): Root dataset directory (e.g., 'data/original_png').
        image_size (tuple): Target image size for resizing (H, W).
        batch_size (int): Batch size.
        split (str): One of 'train', 'validation', 'test'.
        seed (int): Random seed for shuffling.

    Returns:
        tf.data.Dataset: Batched dataset of (images, labels).
        list: Class names in the order used by the dataset.
    """
    split_dir = os.path.join(data_dir, split)

    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    # Determine shuffle based on split
    shuffle = (split == 'train')

    dataset = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels='inferred',
        label_mode='int',           # Integer labels (0, 1, 2, 3)
        class_names=TARGET_CLASSES, # Force consistent class ordering
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,      # Resize to model input size
        shuffle=shuffle,
        seed=seed,
        interpolation='bilinear'
    )

    return dataset


def get_class_weights(data_dir, split='train'):
    """
    Computes class weights from the training set to handle class imbalance.
    Uses inverse frequency weighting: weight_i = total / (num_classes * count_i)

    Parameters:
        data_dir (str): Root dataset directory.
        split (str): Split to compute weights from (default: 'train').

    Returns:
        dict: {class_index: weight} dictionary for model.fit(class_weight=...).
    """
    split_dir = os.path.join(data_dir, split)
    class_counts = {}

    for idx, cls_name in enumerate(TARGET_CLASSES):
        cls_dir = os.path.join(split_dir, cls_name)
        if os.path.exists(cls_dir):
            count = len([f for f in os.listdir(cls_dir) if f.lower().endswith('.png')])
            class_counts[idx] = count
        else:
            class_counts[idx] = 0

    total = sum(class_counts.values())
    num_classes = len(class_counts)

    class_weights = {}
    for idx, count in class_counts.items():
        if count > 0:
            class_weights[idx] = total / (num_classes * count)
        else:
            class_weights[idx] = 1.0

    return class_weights


def get_dataset_info(data_dir):
    """
    Returns a summary dictionary of dataset statistics.

    Returns:
        dict: Contains counts per class per split, total counts, etc.
    """
    info = {}
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(data_dir, split)
        split_info = {}
        for cls_name in TARGET_CLASSES:
            cls_dir = os.path.join(split_dir, cls_name)
            if os.path.exists(cls_dir):
                count = len([f for f in os.listdir(cls_dir) if f.lower().endswith('.png')])
            else:
                count = 0
            split_info[cls_name] = count
        split_info['total'] = sum(split_info.values())
        info[split] = split_info

    return info


def print_dataset_summary(data_dir):
    """Prints a formatted dataset summary table."""
    info = get_dataset_info(data_dir)

    print("\n" + "=" * 65)
    print("  DATASET SUMMARY")
    print("=" * 65)
    print(f"  {'Class':<18} | {'Train':>7} | {'Val':>7} | {'Test':>7} | {'Total':>7}")
    print(f"  {'-'*18}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

    for cls in TARGET_CLASSES:
        tr = info['train'].get(cls, 0)
        va = info['validation'].get(cls, 0)
        te = info['test'].get(cls, 0)
        total = tr + va + te
        print(f"  {cls:<18} | {tr:>7} | {va:>7} | {te:>7} | {total:>7}")

    print(f"  {'TOTAL':<18} | {info['train']['total']:>7} | {info['validation']['total']:>7} | "
          f"{info['test']['total']:>7} | {info['train']['total']+info['validation']['total']+info['test']['total']:>7}")
    print("=" * 65 + "\n")

    return info
