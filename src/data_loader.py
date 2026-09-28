"""
data_loader.py - ICBHI 2017 Respiratory Sound Dataset Loader & Manifest Builder

This script parses the ICBHI 2017 Respiratory Sound Database annotations and diagnosis files,
maps patient IDs to target disease labels, and constructs a balanced manifest DataFrame.

Target Classes (4):
  1. Healthy
  2. Pneumonia
  3. URTI (Upper Respiratory Tract Infection)
  4. Bronchiectasis

Excluded Classes:
  - COPD, Bronchiolitis, Asthma, LRTI (filtered out as per project scope)
"""

import os
import re
import glob
import pandas as pd
import numpy as np

# Target 4 classes requested for classification
TARGET_CLASSES = ['Healthy', 'Pneumonia', 'URTI', 'Bronchiectasis']
CLASS_TO_IDX = {label: idx for idx, label in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS = {idx: label for idx, label in enumerate(TARGET_CLASSES)}

def parse_icbhi_filename(filename):
    """
    Parses ICBHI audio filename format to extract patient ID.
    Filename pattern: {patientID}_{recordingIndex}_{chestLocation}_{acquisitionMode}_{device}.wav
    Example: '101_1b1_Al_sc_Medtron.wav' -> patient_id = 101
    """
    basename = os.path.basename(filename)
    parts = basename.split('_')
    if len(parts) >= 1 and parts[0].isdigit():
        return int(parts[0])
    # Fallback regex if underscore format varies slightly
    match = re.match(r'^(\d+)', basename)
    if match:
        return int(match.group(1))
    return None

def load_diagnosis_mapping(diagnosis_csv_path):
    """
    Parses diagnosis mapping file.
    ICBHI provides diagnosis mapping either as:
    - CSV with columns (Patient_ID/Patient_number, Diagnosis)
    - TXT file formatted as tab-separated values (Patient_ID \t Diagnosis)
    """
    if not os.path.exists(diagnosis_csv_path):
        raise FileNotFoundError(f"Diagnosis file not found at: {diagnosis_csv_path}")

    # Try reading as CSV or whitespace-delimited TXT
    try:
        df = pd.read_csv(diagnosis_csv_path)
    except Exception:
        df = pd.read_csv(diagnosis_csv_path, sep=r'\s+|\t+', engine='python')

    # Standardize column names
    col_mapping = {}
    for col in df.columns:
        c_lower = str(col).strip().lower().replace(' ', '_')
        if 'patient' in c_lower or (c_lower != '101' and 'id' in c_lower):
            col_mapping[col] = 'patient_id'
        elif 'diag' in c_lower or 'disease' in c_lower or 'condition' in c_lower:
            col_mapping[col] = 'diagnosis'

    # If headerless or custom format, reload with explicit headers
    if len(df.columns) == 2 and 'diagnosis' not in col_mapping.values():
        try:
            df = pd.read_csv(diagnosis_csv_path, header=None, names=['patient_id', 'diagnosis'])
        except Exception:
            df = pd.read_csv(diagnosis_csv_path, sep=r'\s+|\t+', header=None, names=['patient_id', 'diagnosis'], engine='python')
    else:
        df = df.rename(columns=col_mapping)

    # Clean patient ID and diagnosis strings
    df['patient_id'] = pd.to_numeric(df['patient_id'], errors='coerce')
    df = df.dropna(subset=['patient_id'])
    df['patient_id'] = df['patient_id'].astype(int)

    # Standardize diagnosis strings
    df['diagnosis'] = df['diagnosis'].astype(str).str.strip()

    # Map variations in label strings (e.g. 'URTI', 'urti', 'Healthy', 'healthy')
    label_map = {
        'healthy': 'Healthy',
        'pneumonia': 'Pneumonia',
        'urti': 'URTI',
        'bronchiectasis': 'Bronchiectasis'
    }

    df['standardized_diagnosis'] = df['diagnosis'].apply(
        lambda x: label_map.get(x.lower(), x)
    )

    mapping = dict(zip(df['patient_id'], df['standardized_diagnosis']))
    return mapping

def build_dataset_manifest(data_dir, diagnosis_file=None):
    """
    Scans data directory for .wav files, matches patient IDs to diagnosis mapping,
    filters for target 4 classes, and creates a dataset manifest DataFrame.

    Parameters:
        data_dir (str): Path to raw ICBHI data directory containing .wav files.
        diagnosis_file (str): Path to diagnosis.csv or filename_diagnosis.csv.

    Returns:
        pd.DataFrame: Cleaned manifest with columns [filepath, filename, patient_id, label, label_idx]
    """
    if diagnosis_file is None:
        # Search for common diagnosis file names in data_dir
        candidates = [
            os.path.join(data_dir, 'diagnosis.csv'),
            os.path.join(data_dir, 'ICBHI_challenge_diagnosis.txt'),
            os.path.join(data_dir, 'filename_diagnosis.csv'),
            os.path.join(data_dir, 'patient_diagnosis.txt')
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                diagnosis_file = candidate
                break

    if not diagnosis_file or not os.path.exists(diagnosis_file):
        raise FileNotFoundError(
            f"Could not locate a valid diagnosis file in {data_dir}. "
            f"Please ensure diagnosis.csv or ICBHI_challenge_diagnosis.txt exists."
        )

    patient_diag_map = load_diagnosis_mapping(diagnosis_file)

    wav_files = glob.glob(os.path.join(data_dir, '**', '*.wav'), recursive=True)
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in directory: {data_dir}")

    records = []
    for filepath in wav_files:
        filename = os.path.basename(filepath)
        patient_id = parse_icbhi_filename(filename)

        if patient_id is not None and patient_id in patient_diag_map:
            label = patient_diag_map[patient_id]
            # Filter strictly for the 4 target classes
            if label in TARGET_CLASSES:
                records.append({
                    'filepath': os.path.abspath(filepath),
                    'filename': filename,
                    'patient_id': patient_id,
                    'label': label,
                    'label_idx': CLASS_TO_IDX[label]
                })

    df = pd.DataFrame(records)
    return df

def print_class_distribution(manifest_df):
    """
    Prints defense-ready data distribution summaries:
    - Total valid audio recordings matching target classes
    - Number of recordings per class
    - Number of unique patient IDs per class
    """
    print("\n" + "="*60)
    print("      ICBHI 2017 TARGET DATASET CLASS DISTRIBUTION")
    print("="*60)
    print(f"Total Selected Recordings : {len(manifest_df)}")
    print(f"Total Unique Patient IDs  : {manifest_df['patient_id'].nunique()}")
    print("-" * 60)

    summary = manifest_df.groupby('label').agg(
        recordings=('filepath', 'count'),
        unique_patients=('patient_id', 'nunique')
    ).reindex(TARGET_CLASSES).fillna(0)

    summary['percentage'] = (summary['recordings'] / len(manifest_df)) * 100

    print(f"{'Class Name':<18} | {'Recordings':<12} | {'Patients':<10} | {'Share (%)':<10}")
    print("-" * 60)
    for cls in TARGET_CLASSES:
        row = summary.loc[cls]
        print(f"{cls:<18} | {int(row['recordings']):<12} | {int(row['unique_patients']):<10} | {row['percentage']:<10.2f}%")
    print("="*60 + "\n")
    return summary

if __name__ == "__main__":
    # Self-test script when run directly
    sample_data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    if os.path.exists(sample_data_dir):
        try:
            df = build_dataset_manifest(sample_data_dir)
            print_class_distribution(df)
        except Exception as e:
            print(f"Data loader test notice: {e}")
    else:
        print(f"Data directory '{sample_data_dir}' not found. Run generate_synthetic_data.py first for zero-setup demo.")
