"""
train_augmented_models.py -- Train & Evaluate All 4 CNN Models Sequentially on Augmented Data

Trains ResNet50, DenseNet121, MobileNetV2, and EfficientNet-B0 on the
augmented PNG spectrogram dataset, evaluates each on the test set, and
prints a final comparison table.

Usage:
    python train_augmented_models.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.trainer import train_single_model
from src.evaluate import evaluate_model, generate_comparison_table
from src.data_pipeline import load_dataset

# ── Shared Configuration ─────────────────────────────────────────────────────
DATA_DIR = os.path.join('data', 'augmented_png')
MODEL_OUTPUT_DIR = 'models_augmented'
RESULTS_BASE_DIR = os.path.join('results', 'augmented')

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30          # Reduced from 50 — synthetic data converges very fast
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.5
DENSE_UNITS = 256
PATIENCE_EARLY_STOP = 7   # Reduced from 10 — prevents over-training
PATIENCE_LR = 4           # Reduced from 5
SEED = 42

# Models to train in order
MODEL_NAMES = ['resnet50', 'densenet121', 'mobilenetv2', 'efficientnet_b0']


def main():
    print("\n" + "=" * 70)
    print("  TRAINING ALL 4 CNN MODELS ON AUGMENTED DATASET (Phase 2)")
    print("=" * 70)
    print(f"  Dataset       : {DATA_DIR}")
    print(f"  Models        : {', '.join(MODEL_NAMES)}")
    print(f"  Max epochs    : {EPOCHS}")
    print(f"  Early stopping: patience={PATIENCE_EARLY_STOP}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Image size    : {IMAGE_SIZE}")
    print("=" * 70 + "\n")

    all_results = []

    for i, model_name in enumerate(MODEL_NAMES, 1):
        print(f"\n{'#' * 70}")
        print(f"  [{i}/{len(MODEL_NAMES)}] PROCESSING: {model_name.upper()}")
        print(f"{'#' * 70}\n")

        results_dir = os.path.join(RESULTS_BASE_DIR, model_name)

        # Skip if already evaluated
        eval_json = os.path.join(results_dir, f"{model_name}_evaluation_results.json")
        if os.path.exists(eval_json):
            import json
            print(f"  >> {model_name} already evaluated, loading existing results...")
            with open(eval_json, 'r') as f:
                results = json.load(f)
            all_results.append(results)
            print(f"\n  [{i}/{len(MODEL_NAMES)}] {model_name.upper()} -- SKIPPED (already done)")
            continue

        # ── Train ────────────────────────────────────────────────────────
        model, history_dict, train_time, model_info = train_single_model(
            model_name=model_name,
            data_dir=DATA_DIR,
            output_dir=MODEL_OUTPUT_DIR,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            dropout_rate=DROPOUT_RATE,
            dense_units=DENSE_UNITS,
            patience_early_stop=PATIENCE_EARLY_STOP,
            patience_lr=PATIENCE_LR,
            seed=SEED
        )

        # ── Evaluate on test set ─────────────────────────────────────────
        print(f"\nLoading test dataset for {model_name} evaluation...")
        test_ds = load_dataset(DATA_DIR, image_size=IMAGE_SIZE,
                               batch_size=BATCH_SIZE, split='test', seed=SEED)

        results = evaluate_model(
            model=model,
            test_dataset=test_ds,
            model_name=model_name,
            output_dir=results_dir,
            history_dict=history_dict,
            train_time=train_time,
            model_info=model_info
        )

        all_results.append(results)

        print(f"\n  [{i}/{len(MODEL_NAMES)}] {model_name.upper()} -- DONE")

    # ── Final comparison ─────────────────────────────────────────────────
    comparison_path = os.path.join(RESULTS_BASE_DIR, 'model_comparison.json')
    generate_comparison_table(all_results, output_path=comparison_path)

    print("\n" + "=" * 70)
    print("  ALL 4 MODELS TRAINED AND EVALUATED SUCCESSFULLY (AUGMENTED)!")
    print("=" * 70)
    print(f"  Model checkpoints : {MODEL_OUTPUT_DIR}/")
    print(f"  Evaluation results: {RESULTS_BASE_DIR}/")
    print(f"  Comparison JSON   : {comparison_path}")
    print("=" * 70 + "\n")

    return all_results


if __name__ == "__main__":
    main()
