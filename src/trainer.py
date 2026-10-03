"""
trainer.py -- CNN Training Orchestration Module

Handles:
    1. Model training with EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    2. Class-weight-aware training for imbalanced datasets
    3. Training history recording (loss, accuracy, time, best epoch)
    4. Model checkpoint saving per architecture
"""

import os
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger
)

from src.models import build_model, get_model_info
from src.data_pipeline import load_dataset, get_class_weights, print_dataset_summary, TARGET_CLASSES


def train_single_model(model_name, data_dir, output_dir,
                        image_size=(224, 224), batch_size=32, epochs=50,
                        learning_rate=1e-3, dropout_rate=0.5, dense_units=256,
                        patience_early_stop=10, patience_lr=5,
                        seed=42):
    """
    Trains a single CNN model end-to-end.

    Parameters:
        model_name (str): One of 'resnet50', 'densenet121', 'mobilenetv2', 'efficientnet_b0'.
        data_dir (str): Path to PNG dataset root (e.g., 'data/original_png').
        output_dir (str): Where to save model checkpoints, history, etc.
        image_size (tuple): Input image size (H, W).
        batch_size (int): Training batch size.
        epochs (int): Maximum number of epochs.
        learning_rate (float): Initial learning rate.
        dropout_rate (float): Dropout rate in classifier head.
        dense_units (int): Number of units in the Dense layer.
        patience_early_stop (int): EarlyStopping patience.
        patience_lr (int): ReduceLROnPlateau patience.
        seed (int): Random seed.

    Returns:
        model: Trained Keras model.
        history_dict: Training history dictionary.
        train_time: Total training time in seconds.
        model_info: Parameter count information.
    """
    # Set random seed for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)

    print("\n" + "=" * 70)
    print(f"  TRAINING: {model_name.upper()}")
    print("=" * 70)

    # ── 1. Load datasets ────────────────────────────────────────────────
    print("\n[1/5] Loading datasets...")
    train_ds = load_dataset(data_dir, image_size=image_size, batch_size=batch_size,
                            split='train', seed=seed)
    val_ds = load_dataset(data_dir, image_size=image_size, batch_size=batch_size,
                          split='validation', seed=seed)

    print_dataset_summary(data_dir)

    # ── 2. Compute class weights ────────────────────────────────────────
    print("[2/5] Computing class weights...")
    class_weights = get_class_weights(data_dir, split='train')
    print(f"  Class weights: {class_weights}")

    # ── 3. Build model ──────────────────────────────────────────────────
    print(f"\n[3/5] Building {model_name} model...")
    model, base_model = build_model(
        model_name,
        input_shape=(*image_size, 3),
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        dense_units=dense_units
    )

    model_info = get_model_info(model, base_model)
    print(f"  Total parameters     : {model_info['total_params']:,}")
    print(f"  Trainable parameters : {model_info['trainable_params']:,}")
    print(f"  Non-trainable params : {model_info['non_trainable_params']:,}")
    print(f"  Base model layers    : {model_info['base_layers']}")

    # ── 4. Setup callbacks ──────────────────────────────────────────────
    print(f"\n[4/5] Configuring training callbacks...")

    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    checkpoint_path = os.path.join(model_output_dir, f"{model_name}_best.keras")
    csv_log_path = os.path.join(model_output_dir, f"{model_name}_training_log.csv")

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            mode='min'
        ),
        CSVLogger(csv_log_path, separator=',', append=False)
    ]

    print(f"  Checkpoint path : {checkpoint_path}")
    print(f"  CSV log path    : {csv_log_path}")
    print(f"  EarlyStopping   : patience={patience_early_stop}, monitor=val_loss")
    print(f"  ReduceLROnPlateau: patience={patience_lr}, factor=0.5, min_lr=1e-7")

    # ── 5. Train ────────────────────────────────────────────────────────
    print(f"\n[5/5] Training {model_name} for up to {epochs} epochs...")
    print("-" * 70)

    start_time = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    train_time = time.time() - start_time
    history_dict = history.history

    # ── Save training metadata ──────────────────────────────────────────
    best_epoch = np.argmin(history_dict['val_loss']) + 1
    best_val_loss = min(history_dict['val_loss'])
    best_val_acc = history_dict['val_accuracy'][best_epoch - 1]

    training_metadata = {
        'model_name': model_name,
        'total_epochs_run': len(history_dict['loss']),
        'max_epochs': epochs,
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val_loss),
        'best_val_accuracy': float(best_val_acc),
        'final_train_loss': float(history_dict['loss'][-1]),
        'final_train_accuracy': float(history_dict['accuracy'][-1]),
        'training_time_seconds': float(train_time),
        'total_params': model_info['total_params'],
        'trainable_params': model_info['trainable_params'],
        'hyperparameters': {
            'image_size': list(image_size),
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'dropout_rate': dropout_rate,
            'dense_units': dense_units,
            'patience_early_stop': patience_early_stop,
            'patience_lr': patience_lr,
            'optimizer': 'Adam',
            'loss': 'sparse_categorical_crossentropy',
            'seed': seed
        },
        'class_weights': {str(k): float(v) for k, v in class_weights.items()}
    }

    metadata_path = os.path.join(model_output_dir, f"{model_name}_training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(training_metadata, f, indent=4)

    # Save history as JSON too
    history_path = os.path.join(model_output_dir, f"{model_name}_history.json")
    history_serializable = {k: [float(v) for v in vals] for k, vals in history_dict.items()}
    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=4)

    # ── Print summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  {model_name.upper()} TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total epochs run       : {len(history_dict['loss'])}")
    print(f"  Best epoch             : {best_epoch}")
    print(f"  Best val loss          : {best_val_loss:.4f}")
    print(f"  Best val accuracy      : {best_val_acc:.4f}")
    print(f"  Final train loss       : {history_dict['loss'][-1]:.4f}")
    print(f"  Final train accuracy   : {history_dict['accuracy'][-1]:.4f}")
    print(f"  Training time          : {train_time:.2f}s ({train_time/60:.1f} min)")
    print(f"  Total parameters       : {model_info['total_params']:,}")
    print(f"  Trainable parameters   : {model_info['trainable_params']:,}")
    print(f"  Checkpoint saved to    : {checkpoint_path}")
    print(f"  Metadata saved to      : {metadata_path}")
    print("=" * 70 + "\n")

    return model, history_dict, train_time, model_info
