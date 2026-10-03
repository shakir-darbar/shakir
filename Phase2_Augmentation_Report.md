# Phase 2: Data Augmentation Report

## 1. Why are we doing Data Augmentation?
In Phase 1, we established a strict baseline using the real clinical dataset without data leakage. As expected, the models scored realistic clinical accuracies (around 40-50%). We are now performing data augmentation on the **Training Set ONLY** to improve these numbers into the 70-86% range.

**Crucial Constraint: No Leaks!**
We do not augment the validation or test sets. If we augment test data, the model's test scores become synthetically inflated. The test set remains 100% real, unaltered patient data to reflect true clinical accuracy.

## 2. What techniques are we using?
Since we are instructed to strictly keep the existing preprocessing (Mel-spectrogram generation), we are applying **SpecAugment** directly to the generated PNG images. This is the industry standard for audio-based deep learning:
1. **Time Masking (Cutout):** We randomly block out vertical slices of the spectrogram (simulating temporary loss of audio, pauses, or irregular breathing rates).
2. **Frequency Masking:** We randomly block out horizontal slices (simulating missing frequencies, microphone anomalies, or different stethoscope types).
3. **Random Brightness/Contrast Jitter:** We slightly vary the pixel intensity to make the model robust against different recording volume levels.

## 3. Dataset Balancing & Sample Generation
The real dataset suffers from severe class imbalance (e.g., only 35 training samples for Bronchiectasis compared to 169 for Healthy). To fix this, we generate more augmented copies for the minority classes to balance the training data.

### Before Augmentation (Original Dataset)
| Class | Train Samples | Validation Samples | Test Samples |
| :--- | :--- | :--- | :--- |
| Healthy | 169 | 35 | 33 |
| Pneumonia | 138 | 87 | 28 |
| URTI | 130 | 13 | 13 |
| Bronchiectasis | 35 | 20 | 55 |
| **TOTAL** | **472** | **155** | **129** |

### Target After Augmentation (Phase 2 Dataset)
We apply class-specific multipliers to the training set only to balance it around ~300-400 samples per class:
* **Healthy** (x2 copies generated) -> ~338 samples
* **Pneumonia** (x2 copies generated) -> ~276 samples
* **URTI** (x3 copies generated) -> ~390 samples
* **Bronchiectasis** (x10 copies generated) -> ~350 samples

| Class | Train Samples | Validation Samples | Test Samples |
| :--- | :--- | :--- | :--- |
| Healthy | 338 | 35 | 33 |
| Pneumonia | 276 | 87 | 28 |
| URTI | 390 | 13 | 13 |
| Bronchiectasis | 350 | 20 | 55 |
| **TOTAL** | **1,354** | **155** | **129** |

**Conclusion:** 
We are effectively generating **882 new synthetic training samples** using SpecAugment, growing the training dataset from 472 to 1,354 samples. This balanced, robust training data is what will push the CNNs to yield much higher accuracy in Phase 2.
