"""
train.py - Model Training & Fine-Tuning Execution Module

Orchestrates the training workflows for both models:
1. Computes class weights to handle residual class imbalances.
2. Trains Model A (Custom CNN) with EarlyStopping and ModelCheckpointing.
3. Trains Model B (MobileNetV2 Transfer Learning) in 2 phases:
   - Phase 1: Train Head (Base Frozen)
   - Phase 2: Fine-tune top layers at reduced learning rate (1e-5)
4. Logs per-epoch accuracy/loss history and records exact training time.
"""

import os
if 'KERAS_BACKEND' not in os.environ:
    os.environ['KERAS_BACKEND'] = 'tensorflow'

import time
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

from src.models import build_custom_cnn, build_mobilenetv2_transfer, unfreeze_and_compile_mobilenetv2

def calculate_class_weights(y_train):
    """
    Computes class weights from training label distribution using sklearn.
    Counters residual class imbalance during loss calculation.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    print(f"Computed Class Weights: {class_weight_dict}")
    return class_weight_dict

def train_custom_cnn(X_train, y_train, X_val, y_val, checkpoint_dir, epochs=50, batch_size=32):
    """
    Trains Custom ResNet-SE CNN (Model A) from scratch.
    """
    print("\n" + "="*60)
    print("      TRAINING MODEL A: CUSTOM RESNET-SE CNN (FROM SCRATCH)")
    print("="*60)

    # One-hot encode integer labels for categorical crossentropy
    num_classes = len(np.unique(y_train))
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)

    class_weight_dict = calculate_class_weights(y_train)

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_custom_cnn(input_shape=input_shape, num_classes=num_classes)
    model.summary()

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "custom_cnn_best.h5")

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    start_time = time.time()
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    elapsed_time = time.time() - start_time
    print(f"\nCustom ResNet-SE CNN Training Completed in {elapsed_time:.2f} seconds.")

    return model, history.history, elapsed_time

def train_mobilenetv2(X_train, y_train, X_val, y_val, checkpoint_dir, epochs_phase1=20, epochs_phase2=15, batch_size=32):
    """
    Trains Model B: MobileNetV2 Transfer Learning.
    Phase 1: Feature Extraction (Head training with frozen base).
    Phase 2: Fine-Tuning top base layers at low learning rate.
    """
    print("\n" + "="*60)
    print("       TRAINING MODEL B: MOBILENETV2 TRANSFER LEARNING")
    print("="*60)

    num_classes = len(np.unique(y_train))
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)

    class_weight_dict = calculate_class_weights(y_train)
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

    model, base_model = build_mobilenetv2_transfer(input_shape=input_shape, num_classes=num_classes)

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "mobilenetv2_best.h5")

    callbacks_p1 = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    print("\n--- MobileNetV2 Phase 1: Training Classification Head ---")
    start_time = time.time()
    history_p1 = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs_phase1,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks_p1,
        verbose=1
    )

    print("\n--- MobileNetV2 Phase 2: Fine-Tuning Base Model ---")
    model = unfreeze_and_compile_mobilenetv2(model, base_model, fine_tune_at=100, fine_tune_lr=1e-5)

    callbacks_p2 = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    history_p2 = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs_phase2,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks_p2,
        verbose=1
    )
    elapsed_time = time.time() - start_time

    # Combine Phase 1 and Phase 2 history metrics for unified plotting
    combined_history = {}
    for key in history_p1.history.keys():
        combined_history[key] = history_p1.history[key] + history_p2.history[key]

    print(f"\nMobileNetV2 Training Completed in {elapsed_time:.2f} seconds.")
    return model, combined_history, elapsed_time
