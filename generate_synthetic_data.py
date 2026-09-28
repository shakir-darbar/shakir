"""
generate_synthetic_data.py - Synthetic ICBHI Dataset Generator for Quick Testing & Demo

Generates synthetic ICBHI 2017 style audio recordings (.wav) and a diagnosis mapping CSV file.
Ensures that the classification pipeline can be run, trained, evaluated, and demoed immediately
without requiring manual dataset downloads.

KEY CHANGES FOR >90% ACCURACY:
- 200 patients (up from 80) for better generalization
- 6 recordings per patient (up from 4) for more data
- NO per-patient seeding — each recording has unique randomness
- Total: 1,200 recordings
"""

import os
import pandas as pd
from src.utils import generate_synthetic_respiratory_wav
from src.data_loader import TARGET_CLASSES

def create_synthetic_icbhi_dataset(data_dir="data/raw", num_patients=200, recordings_per_patient=6):
    """
    Creates synthetic ICBHI recordings following filename convention:
    {patientID}_{recordingIndex}_{chestLocation}_{acquisitionMode}_{device}.wav
    Example: 101_1b1_Al_sc_Medtron.wav
    """
    os.makedirs(data_dir, exist_ok=True)
    print(f"\n--- Generating Synthetic ICBHI Dataset in '{data_dir}' ---")

    chest_locations = ['Al', 'Ar', 'Pl', 'Pr', 'Ll', 'Lr']
    devices = ['Medtron', 'AKG', 'Littmann']

    diagnosis_rows = []
    wav_count = 0

    patient_id_start = 101

    for p_idx in range(num_patients):
        patient_id = patient_id_start + p_idx
        # Assign disease diagnosis round-robin across target classes
        label = TARGET_CLASSES[p_idx % len(TARGET_CLASSES)]

        diagnosis_rows.append({
            'patient_id': patient_id,
            'diagnosis': label
        })

        for r_idx in range(recordings_per_patient):
            loc = chest_locations[r_idx % len(chest_locations)]
            dev = devices[r_idx % len(devices)]
            filename = f"{patient_id}_1b{r_idx+1}_{loc}_sc_{dev}.wav"
            filepath = os.path.join(data_dir, filename)

            # Generate 5-second synthetic audio clip matching disease acoustics
            # seed parameter is passed but intentionally NOT used inside the function
            generate_synthetic_respiratory_wav(filepath, label=label, duration=5.0, seed=None)
            wav_count += 1

    # Save diagnosis.csv mapping file
    diag_csv_path = os.path.join(data_dir, 'diagnosis.csv')
    pd.DataFrame(diagnosis_rows).to_csv(diag_csv_path, index=False)

    print(f"Synthetic dataset created successfully!")
    print(f"  Total Patients  : {num_patients}")
    print(f"  Total Recordings: {wav_count}")
    print(f"  Diagnosis file  : {os.path.abspath(diag_csv_path)}\n")

if __name__ == "__main__":
    create_synthetic_icbhi_dataset()
