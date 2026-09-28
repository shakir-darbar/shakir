"""
preprocessing.py - Respiratory Audio Preprocessing & Mel-Spectrogram Extraction

This module implements the core signal processing pipeline required for cough & respiratory sound analysis:
1. Audio Resampling (16 kHz, Mono)
2. Silence Trimming (Leading & Trailing)
3. Fixed 5-Second Window Segmentation (with 50% overlap / padding)
4. Peak Amplitude Normalization
5. Log-Mel Spectrogram Feature Extraction (128 Mel bands, decibel scale)
"""

import librosa
import numpy as np

# Audio Signal Configuration Constants
SAMPLE_RATE = 16000      # Standardized 16 kHz sampling rate suitable for respiratory acoustics
DURATION = 5.0           # Fixed duration of each audio segment in seconds
NUM_SAMPLES = int(SAMPLE_RATE * DURATION) # 80,000 samples per 5-second window
HOP_OVERLAP = 0.5        # 50% overlap for segmentation (hop duration = 2.5s = 40,000 samples)

# Mel-Spectrogram Hyperparameters
N_MELS = 128             # Number of frequency bins in the Mel scale
N_FFT = 2048             # FFT window length
HOP_LENGTH = 512         # Hop length in samples (yields ~157 time frames for 5s at 16kHz)

def load_and_preprocess_audio(filepath, target_sr=SAMPLE_RATE, top_db=20):
    """
    Loads raw audio file, resamples to target sample rate (16kHz mono),
    and trims leading and trailing background silence.

    Parameters:
        filepath (str): Path to input .wav audio file.
        target_sr (int): Sampling rate (default: 16,000 Hz).
        top_db (int): Decibel threshold below peak considered as silence to trim.

    Returns:
        np.ndarray: Resampled, mono, trimmed audio signal array.
    """
    # Load audio as single-channel mono at specified sampling rate
    audio, sr = librosa.load(filepath, sr=target_sr, mono=True)

    # Trim leading and trailing silence below top_db threshold
    if len(audio) > 0:
        audio, _ = librosa.effects.trim(audio, top_db=top_db)

    # Safety check for empty audio
    if len(audio) == 0:
        audio = np.zeros(NUM_SAMPLES, dtype=np.float32)

    return audio

def normalize_amplitude(audio_segment):
    """
    Applies Peak Normalization to ensure maximum absolute amplitude equals 1.0.
    Prevents scaling issues across different recording devices and distances.
    """
    max_peak = np.max(np.abs(audio_segment))
    if max_peak > 1e-6:
        return audio_segment / max_peak
    return audio_segment

def segment_audio(audio, num_samples=NUM_SAMPLES, overlap=HOP_OVERLAP):
    """
    Segments a long audio signal into fixed-length 5-second windows (80,000 samples).
    Uses sliding window with 50% overlap. Zero-pads short segments if total duration < 5s.

    Parameters:
        audio (np.ndarray): Audio signal array.
        num_samples (int): Length of each window in samples (80,000).
        overlap (float): Overlap ratio (0.5 = 50%).

    Returns:
        list of np.ndarray: List of normalized 5-second audio segments.
    """
    total_len = len(audio)
    hop = int(num_samples * (1.0 - overlap)) # 40,000 samples step

    segments = []

    # Case 1: Audio is shorter than target window -> Zero pad to 5 seconds
    if total_len < num_samples:
        pad_len = num_samples - total_len
        padded_audio = np.pad(audio, (0, pad_len), mode='constant', constant_values=0.0)
        norm_segment = normalize_amplitude(padded_audio)
        segments.append(norm_segment)
        return segments

    # Case 2: Audio length >= 5s -> Sliding window with 50% overlap
    start = 0
    while start + num_samples <= total_len:
        segment = audio[start : start + num_samples]
        segments.append(normalize_amplitude(segment))
        start += hop

    # Handle remaining tail if any audio left over and not fully covered
    if start < total_len and len(segments) == 0:
        tail = audio[-num_samples:] if total_len >= num_samples else np.pad(audio[start:], (0, num_samples - (total_len - start)))
        segments.append(normalize_amplitude(tail))

    return segments

def extract_mel_spectrogram(audio_segment, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Converts a 5-second raw audio signal into a 3-Channel Spectrogram Tensor (128, T, 3).

    Channels:
      0: Log-Mel Spectrogram (Static acoustic features)
      1: Delta Feature (1st order temporal derivative / velocity)
      2: Delta-Delta Feature (2nd order temporal derivative / acceleration)
    """
    # Compute Mel-scaled power spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_segment,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )

    # Convert to decibel (dB) scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Compute 1st derivative (Delta) and 2nd derivative (Delta-Delta)
    delta_spec = librosa.feature.delta(log_mel_spec, order=1)
    delta2_spec = librosa.feature.delta(log_mel_spec, order=2)

    # Standardize per channel (Z-score normalization for scale invariance)
    def _standardize(mat):
        mean = np.mean(mat)
        std = np.std(mat) + 1e-6
        return (mat - mean) / std

    log_mel_norm = _standardize(log_mel_spec)
    delta_norm = _standardize(delta_spec)
    delta2_norm = _standardize(delta2_spec)

    # Stack along 3rd channel dimension -> (128, T, 3)
    spec_3ch = np.stack([log_mel_norm, delta_norm, delta2_norm], axis=-1)

    return spec_3ch.astype(np.float32)

def process_single_audio_file(filepath):
    """
    Full preprocessing pipeline for a single audio file:
    Load -> Resample -> Trim -> Segment -> Normalize -> 3-Channel Spectrogram Tensors.

    Returns:
        list of np.ndarray: List of 3D Spectrogram arrays (128, T, 3) for all segments.
    """
    audio = load_and_preprocess_audio(filepath)
    segments = segment_audio(audio)
    spectrograms = [extract_mel_spectrogram(seg) for seg in segments]
    return spectrograms
