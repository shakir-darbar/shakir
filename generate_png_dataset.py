"""
generate_png_dataset.py — Mel-Spectrogram PNG Dataset Generator

Generates the PNG image dataset for CNN classification experiments.
Uses the FROZEN existing preprocessing pipeline (preprocessing.py) and
saves spectrograms as 3-channel RGB PNG images organized by class.

Workflow:
    1. Generate synthetic ICBHI data if data/raw/ is empty
    2. Build dataset manifest using existing data_loader.py
    3. Patient-wise 3-way split: 70% Train / 15% Validation / 15% Test
    4. Run frozen preprocessing → extract 3-channel Mel spectrograms
    5. Save as PNG images: R=Log-Mel, G=Delta, B=Delta²

Output structure:
    data/original_png/
        train/       {Healthy, URTI, Pneumonia, Bronchiectasis}/
        validation/  {Healthy, URTI, Pneumonia, Bronchiectasis}/
        test/        {Healthy, URTI, Pneumonia, Bronchiectasis}/
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import build_dataset_manifest, print_class_distribution, TARGET_CLASSES
from src.preprocessing import load_and_preprocess_audio, segment_audio, extract_mel_spectrogram

# ─── Configuration ───────────────────────────────────────────────────────────
RAW_DATA_DIR = os.path.join("data", "raw")
OUTPUT_DIR = os.path.join("data", "original_png")
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ─── Utility Functions ──────────────────────────────────────────────────────

def spectrogram_to_rgb_png(spec_3ch):
    """
    Converts a 3-channel spectrogram tensor (128, T, 3) into a uint8 RGB image.

    Each channel is independently min-max normalized to the [0, 255] range:
        - Channel 0 (R): Log-Mel Spectrogram (static energy)
        - Channel 1 (G): Delta (1st derivative / velocity)
        - Channel 2 (B): Delta-Delta (2nd derivative / acceleration)

    Parameters:
        spec_3ch (np.ndarray): Shape (128, T, 3), float32, Z-score normalized.

    Returns:
        PIL.Image: RGB image of size (T, 128) — width=time, height=frequency.
    """
    rgb = np.zeros_like(spec_3ch, dtype=np.float64)

    for ch in range(3):
        channel = spec_3ch[:, :, ch].astype(np.float64)
        ch_min = channel.min()
        ch_max = channel.max()
        if ch_max - ch_min > 1e-8:
            rgb[:, :, ch] = (channel - ch_min) / (ch_max - ch_min) * 255.0
        else:
            rgb[:, :, ch] = 0.0

    # Convert to uint8 and create PIL Image
    rgb_uint8 = rgb.astype(np.uint8)

    # The spectrogram is (frequency_bins, time_frames, 3)
    # Flip vertically so low frequencies are at the bottom (standard convention)
    rgb_uint8 = np.flipud(rgb_uint8)

    img = Image.fromarray(rgb_uint8, mode='RGB')
    return img


def patient_wise_3way_split(manifest_df, train_ratio=0.70, val_ratio=0.15,
                            test_ratio=0.15, random_state=42):
    """
    Splits the manifest into Train / Validation / Test by patient_id.
    Uses two successive GroupShuffleSplit operations to guarantee:
      - Zero patient overlap between any pair of sets
      - Approximate target ratios (70/15/15)

    Parameters:
        manifest_df (pd.DataFrame): Must contain 'patient_id' column.
        train_ratio (float): Target fraction for training set.
        val_ratio (float): Target fraction for validation set.
        test_ratio (float): Target fraction for test set.
        random_state (int): Seed for reproducibility.

    Returns:
        train_df, val_df, test_df: DataFrames for each split.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    # Step 1: Split off the test set first
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
    trainval_idx, test_idx = next(gss_test.split(manifest_df, groups=manifest_df['patient_id']))

    trainval_df = manifest_df.iloc[trainval_idx].copy().reset_index(drop=True)
    test_df = manifest_df.iloc[test_idx].copy().reset_index(drop=True)

    # Step 2: Split the remaining (train+val) into train and validation
    # val_ratio relative to trainval = val_ratio / (train_ratio + val_ratio)
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_ratio, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(trainval_df, groups=trainval_df['patient_id']))

    train_df = trainval_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].copy().reset_index(drop=True)

    # ─── Verification ────────────────────────────────────────────────────
    train_patients = set(train_df['patient_id'].unique())
    val_patients = set(val_df['patient_id'].unique())
    test_patients = set(test_df['patient_id'].unique())

    overlap_tv = train_patients & val_patients
    overlap_tt = train_patients & test_patients
    overlap_vt = val_patients & test_patients

    print("\n" + "=" * 65)
    print("     PATIENT-WISE 3-WAY SPLIT VERIFICATION")
    print("=" * 65)
    print(f"Train      : {len(train_df):>5} recordings | {len(train_patients):>3} unique patients")
    print(f"Validation : {len(val_df):>5} recordings | {len(val_patients):>3} unique patients")
    print(f"Test       : {len(test_df):>5} recordings | {len(test_patients):>3} unique patients")
    print("-" * 65)
    print(f"Train & Val  overlap : {len(overlap_tv)} patients (must be 0)")
    print(f"Train & Test overlap : {len(overlap_tt)} patients (must be 0)")
    print(f"Val   & Test overlap : {len(overlap_vt)} patients (must be 0)")
    print("=" * 65)

    assert len(overlap_tv) == 0, "DATA LEAKAGE: Train and Validation share patients!"
    assert len(overlap_tt) == 0, "DATA LEAKAGE: Train and Test share patients!"
    assert len(overlap_vt) == 0, "DATA LEAKAGE: Validation and Test share patients!"

    print("[OK] No patient leakage detected. Split is clean.\n")

    return train_df, val_df, test_df


def generate_pngs_for_split(split_df, split_name, output_base_dir):
    """
    Processes all audio files in a split manifest, runs the frozen preprocessing
    pipeline, and saves each spectrogram window as a PNG image.

    Parameters:
        split_df (pd.DataFrame): Manifest with columns [filepath, patient_id, label].
        split_name (str): One of 'train', 'validation', 'test'.
        output_base_dir (str): Root output directory (e.g., 'data/original_png').

    Returns:
        dict: Per-class PNG count for this split.
        int: Number of failed conversions.
        list: List of (filepath, error_message) for failed files.
    """
    class_counts = defaultdict(int)
    failed = []
    total_generated = 0

    for idx, row in split_df.iterrows():
        filepath = row['filepath']
        label = row['label']
        patient_id = row['patient_id']
        filename_base = os.path.splitext(os.path.basename(filepath))[0]

        # Create class subdirectory
        class_dir = os.path.join(output_base_dir, split_name, label)
        os.makedirs(class_dir, exist_ok=True)

        try:
            # ── Run the FROZEN preprocessing pipeline ──
            audio = load_and_preprocess_audio(filepath)
            segments = segment_audio(audio)

            for seg_idx, segment in enumerate(segments):
                # Extract 3-channel Mel spectrogram (128, T, 3)
                spec_3ch = extract_mel_spectrogram(segment)

                # Convert to RGB PNG
                img = spectrogram_to_rgb_png(spec_3ch)

                # Save PNG with descriptive filename
                # Format: {patientID}_{origFilename}_seg{N}.png
                png_filename = f"{filename_base}_seg{seg_idx:02d}.png"
                png_path = os.path.join(class_dir, png_filename)
                img.save(png_path, format='PNG')

                class_counts[label] += 1
                total_generated += 1

        except Exception as e:
            failed.append((filepath, str(e)))
            print(f"  [FAIL] FAILED: {filepath} -- {e}")

    return dict(class_counts), len(failed), failed


def print_split_report(split_name, class_counts, sample_img_path=None):
    """Prints a formatted report for one split."""
    total = sum(class_counts.values())
    print(f"\n  {split_name.upper()} SET:")
    print(f"  {'Class':<18} | {'PNG Count':>10}")
    print(f"  {'-'*18}-+-{'-'*10}")
    for cls in TARGET_CLASSES:
        count = class_counts.get(cls, 0)
        print(f"  {cls:<18} | {count:>10}")
    print(f"  {'TOTAL':<18} | {total:>10}")

    if sample_img_path and os.path.exists(sample_img_path):
        img = Image.open(sample_img_path)
        print(f"  Image dimensions : {img.size[0]} x {img.size[1]} pixels (W x H)")
        print(f"  Color format     : {img.mode}")
        print(f"  File format      : PNG")


# ─── Main Execution ─────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  STEP 2: MEL-SPECTROGRAM PNG DATASET GENERATION")
    print("=" * 70)

    # ── 2.1: Generate synthetic data if data/raw/ is empty or missing ──
    if not os.path.exists(RAW_DATA_DIR) or len(os.listdir(RAW_DATA_DIR)) == 0:
        print("\n[2.1] data/raw/ is empty or missing. Generating synthetic ICBHI dataset...")
        from generate_synthetic_data import create_synthetic_icbhi_dataset
        create_synthetic_icbhi_dataset(data_dir=RAW_DATA_DIR)
    else:
        wav_count = len([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.wav')])
        print(f"\n[2.1] Found existing data/raw/ with {wav_count} .wav files. Skipping generation.")

    # ── 2.2: Build manifest using existing data_loader ──
    print("\n[2.2] Building dataset manifest using existing data_loader.py...")
    manifest_df = build_dataset_manifest(RAW_DATA_DIR)
    print_class_distribution(manifest_df)

    # ── 2.3: Patient-wise 3-way split (70/15/15) ──
    print("[2.3] Performing patient-wise 3-way split (70% Train / 15% Val / 15% Test)...")
    train_df, val_df, test_df = patient_wise_3way_split(
        manifest_df,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_state=RANDOM_SEED
    )

    # Print per-class distribution for each split
    for split_name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"\n  {split_name} class distribution:")
        for cls in TARGET_CLASSES:
            count = len(split_df[split_df['label'] == cls])
            patients = split_df[split_df['label'] == cls]['patient_id'].nunique()
            print(f"    {cls:<18}: {count:>4} recordings ({patients} patients)")

    # ── 2.4: Generate PNG images ──
    print("\n" + "-" * 70)
    print("[2.4] Generating PNG spectrogram images using frozen preprocessing...")
    print("-" * 70)

    # Clean output directory
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
        print(f"  Cleaned existing output directory: {OUTPUT_DIR}")

    all_results = {}
    all_failed = []

    for split_name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        print(f"\n  Processing {split_name} set ({len(split_df)} recordings)...")
        counts, num_failed, failures = generate_pngs_for_split(split_df, split_name, OUTPUT_DIR)
        all_results[split_name] = counts
        all_failed.extend(failures)

    # ── 2.5: Generate Summary Report ──
    print("\n" + "=" * 70)
    print("  PNG DATASET GENERATION — SUMMARY REPORT")
    print("=" * 70)

    # Find a sample image for dimension reporting
    sample_img = None
    for split in ["train", "validation", "test"]:
        for cls in TARGET_CLASSES:
            cls_dir = os.path.join(OUTPUT_DIR, split, cls)
            if os.path.exists(cls_dir):
                pngs = [f for f in os.listdir(cls_dir) if f.endswith('.png')]
                if pngs:
                    sample_img = os.path.join(cls_dir, pngs[0])
                    break
        if sample_img:
            break

    for split_name in ["train", "validation", "test"]:
        print_split_report(split_name, all_results.get(split_name, {}),
                           sample_img_path=sample_img if split_name == "train" else None)

    # Grand total
    grand_total = sum(sum(counts.values()) for counts in all_results.values())
    print(f"\n  GRAND TOTAL: {grand_total} PNG images generated")

    if all_failed:
        print(f"\n  [WARNING] FAILED CONVERSIONS: {len(all_failed)}")
        for fp, err in all_failed[:10]:  # Show first 10
            print(f"    [FAIL] {fp}: {err}")
    else:
        print(f"\n  [OK] FAILED CONVERSIONS: 0")

    # Image specs
    if sample_img and os.path.exists(sample_img):
        img = Image.open(sample_img)
        print(f"\n  IMAGE SPECIFICATIONS:")
        print(f"    Dimensions    : {img.size[0]} x {img.size[1]} pixels (Width x Height)")
        print(f"    Color Format  : {img.mode} (R=Log-Mel, G=Delta, B=Delta²)")
        print(f"    File Format   : PNG")
        print(f"    Bit Depth     : 8-bit per channel (uint8, 0-255)")

    # Save split metadata for reproducibility
    metadata = {
        'random_seed': RANDOM_SEED,
        'split_ratios': {
            'train': TRAIN_RATIO,
            'validation': VAL_RATIO,
            'test': TEST_RATIO
        },
        'patient_split': {
            'train_patients': sorted(train_df['patient_id'].unique().tolist()),
            'val_patients': sorted(val_df['patient_id'].unique().tolist()),
            'test_patients': sorted(test_df['patient_id'].unique().tolist()),
        },
        'png_counts': all_results,
        'grand_total': grand_total,
        'failed_count': len(all_failed),
        'target_classes': TARGET_CLASSES,
        'preprocessing_params': {
            'sample_rate': 16000,
            'duration_seconds': 5.0,
            'n_mels': 128,
            'n_fft': 2048,
            'hop_length': 512,
            'overlap': 0.5,
            'channels': ['Log-Mel', 'Delta', 'Delta-Delta']
        }
    }

    metadata_path = os.path.join(OUTPUT_DIR, 'dataset_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"\n  Metadata saved to: {os.path.abspath(metadata_path)}")

    print("\n" + "=" * 70)
    print("  STEP 2 COMPLETE — PNG dataset ready for verification.")
    print("=" * 70 + "\n")

    return all_results, all_failed


if __name__ == "__main__":
    main()
