"""
dataset.py - Patient-Wise Data Split & Feature Dataset Pipeline

Manages:
1. Patient-wise Train/Test Splitting via sklearn's GroupShuffleSplit (80% Train, 20% Test)
   to ensure zero data leakage between training and testing patients.
2. Full dataset processing from raw audio to Mel-spectrogram .npy feature arrays.
3. Saving processed training and testing arrays under `data/processed/train/` and `data/processed/test/`.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src.data_loader import build_dataset_manifest, print_class_distribution, TARGET_CLASSES, CLASS_TO_IDX
from src.preprocessing import load_and_preprocess_audio, segment_audio, extract_mel_spectrogram
from src.augmentation import balance_training_set

def patient_wise_split(manifest_df, test_size=0.20, random_state=42):
    """
    Splits manifest by patient_id using GroupShuffleSplit.
    Ensures that no patient's audio clips appear in both training and test sets.

    Parameters:
        manifest_df (pd.DataFrame): DataFrame with columns [filepath, patient_id, label, label_idx].
        test_size (float): Proportion of patient groups allocated to test set (default: 20%).
        random_state (int): Seed for reproducibility.

    Returns:
        train_df (pd.DataFrame): Manifest for training set.
        test_df (pd.DataFrame): Manifest for testing set.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = manifest_df['patient_id'].values

    train_idx, test_idx = next(gss.split(manifest_df, groups=groups))

    train_df = manifest_df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = manifest_df.iloc[test_idx].copy().reset_index(drop=True)

    # Verification: Confirm set of patient IDs in train and test have 0 overlap
    train_patients = set(train_df['patient_id'].unique())
    test_patients = set(test_df['patient_id'].unique())
    overlap = train_patients.intersection(test_patients)

    print("\n" + "="*60)
    print("           PATIENT-WISE TRAIN / TEST SPLIT VERIFICATION")
    print("="*60)
    print(f"Train Recordings: {len(train_df)} | Unique Patients: {len(train_patients)}")
    print(f"Test Recordings : {len(test_df)}  | Unique Patients: {len(test_patients)}")
    print(f"Patient ID Overlap Count: {len(overlap)} (Must be 0 to prevent data leakage)")
    print("="*60 + "\n")

    assert len(overlap) == 0, "DATA LEAKAGE DETECTED! Patient IDs overlap between train and test sets."
    return train_df, test_df

def extract_features_from_manifest(manifest_df, is_training=False):
    """
    Iterates through manifest records, loads raw audio, segments into 5s clips,
    and converts clips to Mel-Spectrogram arrays (or raw segments for training augmentation).

    Returns:
        X (np.ndarray or list): Array of 2D Spectrograms of shape (N, 128, T).
        y (np.ndarray): Integer label array of shape (N,).
        segment_patient_ids (np.ndarray): Patient ID associated with each 5s segment.
    """
    X_list = []
    y_list = []
    patient_ids_list = []

    for idx, row in manifest_df.iterrows():
        filepath = row['filepath']
        label_idx = row['label_idx']
        patient_id = row['patient_id']

        try:
            audio = load_and_preprocess_audio(filepath)
            segments = segment_audio(audio)

            for seg in segments:
                if is_training:
                    # Keep raw 1D segments to allow raw audio augmentations during balancing
                    X_list.append(seg)
                else:
                    # Convert directly to log Mel-Spectrogram
                    spec = extract_mel_spectrogram(seg)
                    X_list.append(spec)

                y_list.append(label_idx)
                patient_ids_list.append(patient_id)
        except Exception as e:
            print(f"Warning: Skipped audio file {filepath} due to processing error: {e}")

    return X_list, np.array(y_list, dtype=np.int32), np.array(patient_ids_list, dtype=np.int32)

def prepare_and_save_dataset(data_dir, output_dir, test_size=0.20):
    """
    End-to-end dataset preparation pipeline:
    1. Parse ICBHI raw data directory & build manifest
    2. GroupShuffleSplit by patient ID (80% Train / 20% Test)
    3. Extract 5s segments & Mel-Spectrogram features
    4. Apply data augmentation & class balancing on training set
    5. Save processed features (X_train.npy, y_train.npy, X_test.npy, y_test.npy) and metadata
    """
    print("\nStep 1: Building ICBHI Dataset Manifest...")
    manifest_df = build_dataset_manifest(data_dir)
    print_class_distribution(manifest_df)

    print("Step 2: Performing Patient-Wise Train/Test Split...")
    train_manifest, test_manifest = patient_wise_split(manifest_df, test_size=test_size)

    print("Step 3: Extracting Features for Training Set...")
    X_train_raw, y_train_raw, train_pids = extract_features_from_manifest(train_manifest, is_training=True)

    print("Step 4: Applying Data Augmentation to Balance Training Classes (600 samples/class)...")
    X_train, y_train = balance_training_set(X_train_raw, y_train_raw, target_samples_per_class=600)

    print("Step 5: Extracting Features for Test Set...")
    X_test_list, y_test, test_pids = extract_features_from_manifest(test_manifest, is_training=False)
    X_test = np.array(X_test_list, dtype=np.float32)

    # Ensure shape format is (N, 128, T)
    if X_train.ndim == 3:
        # X_train is already list of spectrograms
        pass
    else:
        # Convert raw audio clips to spectrograms if not already converted
        X_train_specs = [extract_mel_spectrogram(s) if s.ndim == 1 else s for s in X_train]
        X_train = np.array(X_train_specs, dtype=np.float32)

    # Ensure 4D tensor shape format (N, 128, T, C)
    if X_train.ndim == 3:
        X_train_expanded = np.expand_dims(X_train, axis=-1)
    else:
        X_train_expanded = X_train

    if X_test.ndim == 3:
        X_test_expanded = np.expand_dims(X_test, axis=-1)
    else:
        X_test_expanded = X_test

    print("\n--- Processed Feature Shapes ---")
    print(f"X_train shape: {X_train_expanded.shape} | y_train shape: {y_train.shape}")
    print(f"X_test shape : {X_test_expanded.shape}  | y_test shape : {y_test.shape}")

    # Create output folders
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

    # Save feature numpy arrays
    np.save(os.path.join(output_dir, 'train', 'X_train.npy'), X_train_expanded)
    np.save(os.path.join(output_dir, 'train', 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'test', 'X_test.npy'), X_test_expanded)
    np.save(os.path.join(output_dir, 'test', 'y_test.npy'), y_test)

    # Save class label mapping metadata
    label_map_path = os.path.join(output_dir, 'label_mapping.json')
    with open(label_map_path, 'w') as f:
        json.dump({'target_classes': TARGET_CLASSES, 'class_to_idx': CLASS_TO_IDX}, f, indent=4)

    print(f"\nFeature dataset successfully saved to: {output_dir}\n")
    return X_train_expanded, y_train, X_test_expanded, y_test
