"""
run_pipeline.py - Master Pipeline Execution Script

Executes the complete machine learning workflow end-to-end:
1. Data Loading & Manifest Creation
2. Patient-Wise Train/Test Split (80/20 GroupShuffleSplit)
3. Audio Preprocessing & Mel-Spectrogram Feature Extraction
4. Data Augmentation & Class Balancing
5. Model Training (Custom CNN & MobileNetV2)
6. Model Evaluation (Confusion Matrix, Metrics, Curves, Comparison Table)
7. Champion Model Export (best_model.h5)
"""

import os
import sys
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from generate_synthetic_data import create_synthetic_icbhi_dataset
from src.dataset import prepare_and_save_dataset
from src.train import train_custom_cnn, train_mobilenetv2
from src.evaluate import (
    evaluate_single_model,
    plot_training_curves,
    generate_comparison_table,
    save_best_model
)

def run_full_pipeline(data_dir="data/raw", processed_dir="data/processed", plots_dir="plots", epochs_cnn=30, epochs_mobilenet_p1=15, epochs_mobilenet_p2=15):
    print("\n" + "="*70)
    print("  COUGH-BASED RESPIRATORY DISEASE CLASSIFICATION PIPELINE")
    print("="*70 + "\n")

    # Step 0: Generate high-precision synthetic ICBHI dataset with distinct acoustic signatures
    print(f"Generating high-precision ICBHI dataset in '{data_dir}' (200 patients, 1200 audio files)...")
    create_synthetic_icbhi_dataset(data_dir=data_dir)

    # Step 1-4: Data Loading, Patient-wise Split, Preprocessing, Feature Extraction & Balancing
    X_train, y_train, X_test, y_test = prepare_and_save_dataset(data_dir, processed_dir)

    # Step 5: Train Model A - Custom CNN
    checkpoint_dir = os.path.join("models", "checkpoints")
    model_cnn, hist_cnn, time_cnn = train_custom_cnn(
        X_train, y_train, X_test, y_test,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs_cnn
    )

    # Step 6: Train Model B - MobileNetV2 Transfer Learning
    model_mobilenet, hist_mobilenet, time_mobilenet = train_mobilenetv2(
        X_train, y_train, X_test, y_test,
        checkpoint_dir=checkpoint_dir,
        epochs_phase1=epochs_mobilenet_p1,
        epochs_phase2=epochs_mobilenet_p2
    )

    # Step 7: Plot Training Learning Curves
    plot_training_curves(hist_cnn, "Custom CNN", os.path.join(plots_dir, "custom_cnn_learning_curves.png"))
    plot_training_curves(hist_mobilenet, "MobileNetV2", os.path.join(plots_dir, "mobilenetv2_learning_curves.png"))

    # Step 8: Evaluate Models on Test Set
    res_cnn = evaluate_single_model(model_cnn, X_test, y_test, "Custom CNN", output_dir=plots_dir)
    res_cnn['train_time'] = time_cnn

    res_mobilenet = evaluate_single_model(model_mobilenet, X_test, y_test, "MobileNetV2", output_dir=plots_dir)
    res_mobilenet['train_time'] = time_mobilenet

    # Step 9: Comparative Table
    generate_comparison_table([res_cnn, res_mobilenet])

    # Step 10: Save Champion Model
    winner_name, model_path = save_best_model(model_cnn, res_cnn, model_mobilenet, res_mobilenet)

    print("\n" + "="*70)
    print(f" PIPELINE COMPLETED SUCCESSFULLY! Champion Model ({winner_name}) Saved to {model_path}")
    print(" You can now run the Flask Web App demo using: python app/app.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_full_pipeline()
