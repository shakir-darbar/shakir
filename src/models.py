"""
models.py - Model Architectures (Custom CNN & MobileNetV2 Transfer Learning)

Defines deep learning model architectures for respiratory sound classification:
1. Model A: Custom 2D Convolutional Neural Network (built from scratch)
   - 3 Convolutional Blocks: [Conv2D -> BatchNorm -> ReLU -> MaxPooling2D]
   - Feature Aggregation: GlobalAveragePooling2D
   - Fully Connected Head: Dense(128, ReLU) -> Dropout(0.5) -> Dense(4, Softmax)

2. Model B: Transfer Learning Baseline using MobileNetV2 (Pretrained on ImageNet)
   - Input channel duplication (1-channel Mel-Spectrogram -> 3-channel RGB image format)
   - MobileNetV2 feature extractor (Base frozen during initial head training)
   - Fine-tuning capability for top layers with reduced learning rate (1e-5)
"""

import os
if 'KERAS_BACKEND' not in os.environ:
    os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras import layers, models, optimizers

NUM_CLASSES = 4

def squeeze_and_excitation_block(input_tensor, ratio=16, name_prefix="se"):
    """
    Squeeze-and-Excitation (SE) Channel Attention Module.
    Auto-weighs frequency channel importance.
    """
    init = input_tensor
    filters = init.shape[-1]
    se = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(init)
    se = layers.Dense(max(4, filters // ratio), activation='relu', name=f"{name_prefix}_sq")(se)
    se = layers.Dense(filters, activation='sigmoid', name=f"{name_prefix}_ex")(se)
    se = layers.Reshape((1, 1, filters), name=f"{name_prefix}_reshape")(se)
    return layers.Multiply(name=f"{name_prefix}_mult")([init, se])

def residual_block(x, filters, kernel_size=(3, 3), block_id=1):
    """
    Residual Convolutional Block with Skip Connection and Channel Attention.
    """
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same', name=f"res{block_id}_proj")(shortcut)
        shortcut = layers.BatchNormalization(name=f"res{block_id}_proj_bn")(shortcut)

    x = layers.Conv2D(filters, kernel_size, padding='same', name=f"res{block_id}_conv1")(x)
    x = layers.BatchNormalization(name=f"res{block_id}_bn1")(x)
    x = layers.Activation('relu', name=f"res{block_id}_relu1")(x)

    x = layers.Conv2D(filters, kernel_size, padding='same', name=f"res{block_id}_conv2")(x)
    x = layers.BatchNormalization(name=f"res{block_id}_bn2")(x)

    x = squeeze_and_excitation_block(x, ratio=8, name_prefix=f"res{block_id}_se")

    x = layers.Add(name=f"res{block_id}_add")([shortcut, x])
    x = layers.Activation('relu', name=f"res{block_id}_relu2")(x)
    x = layers.MaxPooling2D((2, 2), name=f"res{block_id}_pool")(x)
    return x

def build_custom_cnn(input_shape=(128, 157, 3), num_classes=NUM_CLASSES, learning_rate=1e-3):
    """
    Builds Model A: High-Performance Custom ResNet-SE 2D CNN Architecture.
    Specifically engineered for 3-Channel Mel-Delta-Delta Spectrogram Tensors.
    """
    inputs = layers.Input(shape=input_shape, name="input_spectrogram_3ch")

    # Stem Block
    x = layers.Conv2D(32, (5, 5), padding='same', name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation('relu', name="stem_relu")(x)

    # Residual SE Blocks (3 blocks — reduced from 4 to prevent overfitting)
    x = residual_block(x, filters=32, block_id=1)
    x = residual_block(x, filters=64, block_id=2)
    x = residual_block(x, filters=128, block_id=3)

    # Global Average Pooling & Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation='relu', name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Dropout(0.45, name="dropout1")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Custom_ResNet_SE_Respiratory")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )

    return model

def build_mobilenetv2_transfer(input_shape=(128, 157, 3), num_classes=NUM_CLASSES, learning_rate=1e-3):
    """
    Builds Model B: MobileNetV2 Transfer Learning Model on 3-Channel Spectrograms.
    """
    inputs = layers.Input(shape=input_shape, name="input_spectrogram_3ch")

    # MobileNetV2 Backbone pretrained on ImageNet
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = base_model(inputs, training=False)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="mobilenet_gap")(x)
    x = layers.Dense(256, activation='relu', name="mobilenet_fc1")(x)
    x = layers.BatchNormalization(name="mobilenet_bn")(x)
    x = layers.Dropout(0.45, name="mobilenet_dropout")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="mobilenet_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Transfer_Respiratory")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )

    return model, base_model

def unfreeze_and_compile_mobilenetv2(model, base_model, fine_tune_at=100, fine_tune_lr=1e-5):
    """
    Unfreezes the top layers of MobileNetV2 for phase 2 fine-tuning with a lower learning rate.
    """
    base_model.trainable = True

    # Freeze all layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=fine_tune_lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    print(f"\nMobileNetV2 Base Unfrozen from Layer {fine_tune_at}. Fine-tuning LR set to {fine_tune_lr}")
    return model
