"""
augmentation.py - Audio and Spectrogram Data Augmentation Pipeline

Implements domain-specific data augmentations for respiratory sound classification:
1. Raw Audio Level (applied before spectrogram conversion):
   - Time-Stretching (speed up / slow down sound without changing pitch)
   - Pitch-Shifting (transpose pitch by semitones without changing duration)
   - Background Noise Injection (Gaussian noise addition to simulate clinical environment noise)
2. Spectrogram Level (applied to Mel-Spectrogram matrix):
   - SpecAugment Frequency Masking (zero out random frequency bands)
   - SpecAugment Time Masking (zero out random time channels)

Class Balancing:
   - Oversamples minority classes in training data using dynamic augmentation
     until all target classes reach equal sample representation.
"""

import numpy as np
import librosa
from src.preprocessing import extract_mel_spectrogram, SAMPLE_RATE

def augment_time_stretch(audio_segment, rate_range=(0.85, 1.15)):
    """
    Stretches or compresses audio in time domain without altering pitch.
    """
    rate = np.random.uniform(rate_range[0], rate_range[1])
    try:
        stretched = librosa.effects.time_stretch(audio_segment, rate=rate)
        # Ensure fixed output length equal to original input length
        if len(stretched) > len(audio_segment):
            return stretched[:len(audio_segment)]
        else:
            return np.pad(stretched, (0, len(audio_segment) - len(stretched)))
    except Exception:
        return audio_segment

def augment_pitch_shift(audio_segment, sr=SAMPLE_RATE, n_steps_range=(-2, 2)):
    """
    Shifts pitch up or down by random semitones without altering duration.
    """
    n_steps = np.random.randint(n_steps_range[0], n_steps_range[1] + 1)
    if n_steps == 0:
        return audio_segment
    try:
        shifted = librosa.effects.pitch_shift(audio_segment, sr=sr, n_steps=n_steps)
        return shifted
    except Exception:
        return audio_segment

def augment_noise_injection(audio_segment, noise_factor=0.005):
    """
    Injects random Gaussian white noise into raw audio.
    Simulates ambient room acoustics and microphone noise.
    """
    noise = np.random.normal(0, 1, len(audio_segment))
    augmented = audio_segment + noise_factor * noise
    return augmented

def apply_spec_augment(spectrogram, max_mask_freq=16, max_mask_time=20, num_freq_masks=1, num_time_masks=1):
    """
    SpecAugment (Park et al., 2019):
    Applies frequency and time masking directly to 2D or 3D Spectrogram tensors.
    """
    spec = spectrogram.copy()
    if spec.ndim == 2:
        n_mels, num_steps = spec.shape
    else:
        n_mels, num_steps, _ = spec.shape
    fill_value = float(spec.min())

    # Frequency Masking
    for _ in range(num_freq_masks):
        f = np.random.randint(0, max_mask_freq)
        f0 = np.random.randint(0, max(1, n_mels - f))
        if spec.ndim == 2:
            spec[f0 : f0 + f, :] = fill_value
        else:
            spec[f0 : f0 + f, :, :] = fill_value

    # Time Masking
    for _ in range(num_time_masks):
        t = np.random.randint(0, max_mask_time)
        t0 = np.random.randint(0, max(1, num_steps - t))
        if spec.ndim == 2:
            spec[:, t0 : t0 + t] = fill_value
        else:
            spec[:, t0 : t0 + t, :] = fill_value

    return spec

def augment_gain_scaling(audio_segment, scale_range=(0.6, 1.4)):
    """
    Applies random amplitude gain scaling to simulate soft/loud recordings.
    """
    gain = np.random.uniform(scale_range[0], scale_range[1])
    return audio_segment * gain

def generate_augmented_spectrogram(audio_segment):
    """
    Applies a random combination of raw audio transforms and SpecAugment
    to generate a unique synthetic sample from an existing 5-second audio clip.
    """
    augmented_audio = audio_segment.copy()

    # Random choice of audio domain augmentations
    if np.random.rand() > 0.5:
        augmented_audio = augment_time_stretch(augmented_audio)
    if np.random.rand() > 0.5:
        augmented_audio = augment_pitch_shift(augmented_audio)
    if np.random.rand() > 0.4:
        augmented_audio = augment_noise_injection(augmented_audio)
    if np.random.rand() > 0.4:
        augmented_audio = augment_gain_scaling(augmented_audio)

    # Convert augmented audio to 3-Channel Mel-Spectrogram
    spec = extract_mel_spectrogram(augmented_audio)

    # Apply SpecAugment to final spectrogram
    if np.random.rand() > 0.3:
        spec = apply_spec_augment(spec, max_mask_freq=20, max_mask_time=24)

    return spec

def balance_training_set(X_train_raw, y_train, target_samples_per_class=400):
    """
    Balances minority and majority classes in the training set by applying dynamic audio/spec augmentations
    until each of the target 4 classes has exactly `target_samples_per_class` samples (default: 400 per class).

    Parameters:
        X_train_raw (list or np.ndarray): Training raw audio segments or 3D spectrograms.
        y_train (np.ndarray): Training labels (integer indices).
        target_samples_per_class (int): Target number of samples per class (default: 400).

    Returns:
        X_balanced (np.ndarray): Balanced spectrogram feature array of shape (N, 128, T, 3).
        y_balanced (np.ndarray): Balanced label array of shape (N,).
    """
    classes, counts = np.unique(y_train, return_counts=True)
    max_count = target_samples_per_class if target_samples_per_class is not None else np.max(counts)

    # Ensure baseline training samples are converted to 3D Mel-Spectrograms
    X_specs = []
    for sample in X_train_raw:
        if sample.ndim == 1:
            X_specs.append(extract_mel_spectrogram(sample))
        else:
            X_specs.append(sample)

    X_balanced_list = []
    y_balanced_list = []

    print(f"\n--- Applying Enhanced Data Augmentation (Target: {target_samples_per_class} Samples / Class) ---")
    for cls in classes:
        cls_indices = np.where(y_train == cls)[0]
        cls_count = len(cls_indices)

        # Include original samples up to max_count
        selected_orig_count = min(cls_count, max_count)
        for idx in cls_indices[:selected_orig_count]:
            sample = X_train_raw[idx]
            spec = extract_mel_spectrogram(sample) if sample.ndim == 1 else sample
            X_balanced_list.append(spec)
            y_balanced_list.append(cls)

        needed = max_count - selected_orig_count
        print(f"Class {cls}: Initial = {cls_count} samples. Selected original = {selected_orig_count}. Augmenting {needed} synthetic samples...")

        if needed > 0:
            for i in range(needed):
                # Pick a random sample from this class to augment
                src_idx = np.random.choice(cls_indices)
                sample = X_train_raw[src_idx]

                # If sample is raw audio segment (1D), apply raw audio transforms + spec
                if sample.ndim == 1:
                    aug_spec = generate_augmented_spectrogram(sample)
                else:
                    aug_spec = apply_spec_augment(sample)

                X_balanced_list.append(aug_spec)
                y_balanced_list.append(cls)

    X_balanced = np.array(X_balanced_list, dtype=np.float32)
    y_balanced = np.array(y_balanced_list, dtype=np.int32)

    # Shuffle balanced dataset
    shuffle_idx = np.random.permutation(len(y_balanced))
    return X_balanced[shuffle_idx], y_balanced[shuffle_idx]
