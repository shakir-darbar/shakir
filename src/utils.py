"""
utils.py - Utility Functions & Synthetic ICBHI Generator Helpers

Provides:
- Directory creation helpers
- Audio frequency generator utilities (synthesizing realistic cough/respiratory sound patterns)

CRITICAL DESIGN DECISION:
  - NO per-patient seeding is used. Each recording gets fresh randomness.
  - Disease-specific spectral signatures are DOMINANT (5-10x stronger than noise).
  - This ensures the model learns disease features, NOT patient identity.
"""

import os
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from src.data_loader import TARGET_CLASSES

def _butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(0.001, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def generate_synthetic_respiratory_wav(output_path, label, duration=5.0, sr=16000, seed=None):
    """
    Generates synthetic audio wave with HIGHLY DISTINCT and DOMINANT disease spectral signatures.

    Key principle: Disease signal >> noise, so that Mel-spectrograms are trivially
    separable even across different patients.

    Classes:
      - Healthy: Soft vesicular breathing — gentle bandpassed pink noise (100-400 Hz)
                 with slow respiratory cycle modulation. QUIET overall.
      - Pneumonia: Dense crackling bursts — many high-frequency transient spikes (1000-4000 Hz)
                   with strong energy above 1 kHz. Very distinctive high-freq texture.
      - URTI: Explosive cough bursts — strong harmonic tones (250/500/750/1000 Hz)
              with sharp exponential decay. Clear tonal events.
      - Bronchiectasis: Continuous polyphonic wheezing — strong sustained sinusoidal tones
                        (500/800/1200 Hz) with amplitude modulation. Continuous tonal energy.

    NOTE: seed parameter is accepted but NOT USED to prevent patient fingerprinting.
    """
    # INTENTIONALLY do NOT set np.random.seed — each recording gets unique randomness
    # This prevents the model from learning patient-specific noise patterns

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    if label == 'Healthy':
        # Soft vesicular breathing: gentle bandpassed noise with respiratory cycle
        noise = np.random.normal(0, 0.03, len(t))  # Very soft noise
        sos = _butter_bandpass(100, 400, sr, order=4)
        filtered_noise = sosfilt(sos, noise)
        # Slow breathing cycle (0.25-0.35 Hz, randomized)
        breath_rate = np.random.uniform(0.25, 0.35)
        resp_envelope = 0.5 * (1.0 + np.sin(2 * np.pi * breath_rate * t - np.pi / 2))
        audio = filtered_noise * (resp_envelope + 0.05)
        # Keep it quiet — healthy lungs are soft
        audio *= 0.3

    elif label == 'Pneumonia':
        # Dense high-frequency crackles: explosive transient bursts above 1 kHz
        # Background: very faint low-freq breathing
        bg_noise = np.random.normal(0, 0.01, len(t))
        sos_bg = _butter_bandpass(100, 400, sr, order=3)
        bg = sosfilt(sos_bg, bg_noise) * 0.05

        # STRONG crackle events: 100-150 spikes spread throughout
        crackle_signal = np.zeros_like(t)
        num_spikes = np.random.randint(100, 160)
        spike_indices = np.random.choice(len(t), size=num_spikes, replace=False)
        crackle_signal[spike_indices] = np.random.uniform(2.0, 5.0, size=num_spikes)

        # Additional clustered bursts (3-6 burst regions)
        num_bursts = np.random.randint(3, 7)
        for _ in range(num_bursts):
            burst_center = np.random.randint(int(0.1 * len(t)), int(0.9 * len(t)))
            burst_width = np.random.randint(200, 800)
            start_b = max(0, burst_center - burst_width // 2)
            end_b = min(len(t), burst_center + burst_width // 2)
            burst_spikes = np.random.choice(range(start_b, end_b), size=min(30, end_b - start_b), replace=False)
            crackle_signal[burst_spikes] += np.random.uniform(1.5, 4.0, size=len(burst_spikes))

        # Bandpass crackles to high frequency (1000-4000 Hz)
        sos_crackle = _butter_bandpass(1000, 4000, sr, order=4)
        filtered_crackles = sosfilt(sos_crackle, crackle_signal)

        audio = bg + filtered_crackles * 2.0  # Crackles DOMINATE

    elif label == 'URTI':
        # Explosive cough bursts with strong harmonic content
        cough_signal = np.zeros_like(t)

        # 3-5 cough bursts
        num_coughs = np.random.randint(3, 6)
        burst_starts = np.sort(np.random.uniform(0.3, duration - 0.6, num_coughs))

        for b_start in burst_starts:
            cough_dur = np.random.uniform(0.2, 0.5)
            mask = (t >= b_start) & (t <= b_start + cough_dur)
            t_sub = t[mask] - b_start

            # Sharp exponential decay envelope
            env = np.exp(-12 * t_sub)

            # Strong harmonic triad with randomized fundamental (220-350 Hz)
            f0 = np.random.uniform(220, 350)
            harmonics = (
                1.0 * np.sin(2 * np.pi * f0 * t_sub) +
                0.8 * np.sin(2 * np.pi * 2 * f0 * t_sub) +
                0.5 * np.sin(2 * np.pi * 3 * f0 * t_sub) +
                0.3 * np.sin(2 * np.pi * 4 * f0 * t_sub)
            )
            cough_signal[mask] += harmonics * env * 1.5  # Strong amplitude

        # Very faint background noise
        bg_noise = np.random.normal(0, 0.005, len(t))
        audio = cough_signal + bg_noise

    elif label == 'Bronchiectasis':
        # Continuous polyphonic wheezing: sustained tonal sinusoids
        # Primary wheeze tones (randomized slightly per recording)
        f1 = np.random.uniform(480, 550)
        f2 = np.random.uniform(780, 850)
        f3 = np.random.uniform(1150, 1300)

        # Strong sustained tones with FM vibrato
        wheeze1 = 0.7 * np.sin(2 * np.pi * f1 * t + 0.4 * np.sin(2 * np.pi * 2.5 * t))
        wheeze2 = 0.5 * np.sin(2 * np.pi * f2 * t + 0.3 * np.sin(2 * np.pi * 3.0 * t))
        wheeze3 = 0.3 * np.sin(2 * np.pi * f3 * t)

        # Amplitude modulation: slow respiratory cycle
        am_rate = np.random.uniform(0.25, 0.4)
        am_envelope = 0.6 + 0.4 * np.sin(2 * np.pi * am_rate * t)

        # Very soft background bubbling
        bg_noise = np.random.normal(0, 0.01, len(t))
        sos_bubble = _butter_bandpass(150, 700, sr, order=3)
        bubbling = sosfilt(sos_bubble, bg_noise) * 0.1

        audio = (wheeze1 + wheeze2 + wheeze3) * am_envelope + bubbling
    else:
        audio = np.random.normal(0, 0.01, len(t))

    # Peak normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = (audio / max_val) * 0.95

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio.astype(np.float32), sr)
    return output_path
