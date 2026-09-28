# Cough-Based Respiratory Disease Classification Using Deep Learning

A complete, end-to-end Python deep learning solution for classifying respiratory audio recordings into **4 target disease classes**:
1. **Healthy**
2. **Pneumonia**
3. **URTI** (Upper Respiratory Tract Infection)
4. **Bronchiectasis**

Built using the **ICBHI 2017 Respiratory Sound Database**, comparing a **Custom 2D CNN** (built from scratch) against a **MobileNetV2 Transfer Learning** baseline, and wrapped in a modern **Flask Web Application** for interactive project demonstration.

---

## 📁 Project Directory Structure

```text
Cough-Based-Respiratory-Disease-Classification/
│
├── data/
│   ├── raw/                   # Place real ICBHI .wav, .txt, diagnosis.csv here
│   └── processed/             # Output .npy spectrograms & train/test matrices
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # ICBHI filename parser, diagnosis mapping & 4-class filtering
│   ├── preprocessing.py       # Resample (16kHz mono), trim, 5s windowing, peak norm, Mel-spec
│   ├── augmentation.py        # Time-stretch, pitch-shift, noise injection, SpecAugment & class balancing
│   ├── dataset.py             # Patient-wise GroupShuffleSplit (80/20) & feature set generator
│   ├── models.py              # Custom CNN & MobileNetV2 architecture builders
│   ├── train.py               # Model training script with EarlyStopping, Checkpointing & class weights
│   ├── evaluate.py            # Confusion matrix heatmaps, learning curves, metrics report & comparison table
│   └── utils.py               # Audio signal synthesizer & helper utilities
│
├── app/
│   ├── app.py                 # Flask web server (/predict API endpoint)
│   ├── templates/
│   │   └── index.html         # Modern dark-mode web interface
│   └── static/
│       ├── css/
│       │   └── style.css      # Glassmorphism design tokens & styles
│       └── js/
│           └── main.js        # File upload handling, audio player, AJAX & dynamic charts
│
├── plots/                     # Output evaluation plots (confusion matrices & training curves)
├── generate_synthetic_data.py # Quick synthetic dataset generator for zero-setup demo
├── run_pipeline.py            # Master CLI runner (data -> features -> train -> evaluate)
├── best_model.h5              # Champion saved model weights
├── label_mapping.json         # Class mapping metadata
├── requirements.txt           # Python package dependencies
└── README.md                  # Complete project & defense documentation
```

---

## ⚡ Quickstart Guide (Zero-Setup Execution)

You can test and run the entire pipeline immediately, even before downloading the real ICBHI dataset files!

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full Machine Learning Pipeline
```bash
python run_pipeline.py
```
*Note: If `data/raw/` is empty, `run_pipeline.py` automatically generates a synthetic ICBHI dataset, builds the features, trains both models, plots learning curves, generates confusion matrices, prints the comparison table, and exports `best_model.h5`.*

### 3. Launch the Demo Web Application
```bash
python app/app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## 🔬 Using the Real ICBHI 2017 Dataset

To train on the official **ICBHI 2017 Respiratory Sound Database**:

1. Download the dataset files from Kaggle or official ICBHI source.
2. Copy all `.wav` files, `.txt` annotation files, and `diagnosis.csv` (or `ICBHI_challenge_diagnosis.txt`) into **`data/raw/`**.
3. Run the master pipeline script:
   ```bash
   python run_pipeline.py
   ```
4. The system will parse patient IDs from filenames (`{patientID}_{recordingIndex}_{chestLocation}_{acquisitionMode}_{device}.wav`), filter strictly for the 4 target classes (**Healthy**, **Pneumonia**, **URTI**, **Bronchiectasis**), discard unwanted classes (COPD, Bronchiolitis, Asthma, LRTI), perform patient-wise splitting, augment training samples, train both models, and save the best model to `best_model.h5`.

---

## 🎓 Defense Q&A Cheatsheet (Key Concepts for Project Defense)

### 1. Why 16 kHz Mono Resampling & Silence Trimming?
- **16 kHz Sampling Rate**: Respiratory sounds (breath sounds, crackles, wheezes, coughs) contain primary acoustic power between $100\text{ Hz}$ and $4000\text{ Hz}$. According to the Nyquist-Shannon sampling theorem, a sampling rate of $16\text{ kHz}$ captures frequencies up to $8\text{ kHz}$, preserving all critical diagnostic frequency components while saving computational memory.
- **Silence Trimming**: Removes non-informative leading and trailing background silence (`top_db=20`), preventing the neural network from learning room background noise patterns instead of respiratory features.

### 2. Why 5-Second Windowing with 50% Overlap?
- Acoustic signals vary in length. Deep learning CNNs require uniform input tensor shapes.
- 5 seconds ($80,000\text{ samples}$ at $16\text{ kHz}$) is optimal to capture at least one full breath cycle or cough episode.
- **50% Overlap (2.5s hop)** ensures adventitious sounds (crackles/wheezes) occurring at segment boundaries are captured in at least one window.

### 3. Why Log-Mel Spectrograms over Raw Waveforms or MFCCs?
- **Human Auditory Perception (Mel Scale)**: Projects linear Hertz frequencies onto non-linear Mel frequency bands ($128$ bands), giving higher resolution to lower frequencies where breath acoustics dominate.
- **Log (Decibel) Scale**: Converts raw power to decibels (`librosa.power_to_db`), compressing high dynamic range and matching human loudness perception.
- **2D Spatial Representation**: Allows 2D Convolutional Neural Networks (CNNs) to apply spatial filters across time and frequency dimensions simultaneously.

### 4. How is Data Leakage Prevented?
- Split by **Patient ID** using `sklearn.model_selection.GroupShuffleSplit` (80% Train / 20% Test).
- **Critical Rule**: All segments belonging to patient $X$ are placed strictly in either the training set OR the testing set. If segments from the same patient appeared in both sets, the model would memorize patient-specific vocal tract characteristics rather than disease signatures.

### 5. What Data Augmentations are Applied?
- **Audio Domain**: Time-stretching, pitch-shifting, and Gaussian noise injection.
- **Spectrogram Domain**: SpecAugment (frequency and time channel masking).
- **Class Balancing**: Minority classes are dynamically augmented until all 4 target classes reach equal sample counts in the training set.

### 6. Model Architectures Comparison
| Feature / Metric | Model A: Custom CNN | Model B: MobileNetV2 Transfer Learning |
| :--- | :--- | :--- |
| **Origin** | Built from scratch for spectrograms | Pretrained on ImageNet ($1.4\text{M}$ natural images) |
| **Input Shape** | $(128, 157, 1)$ | $(128, 157, 3)$ (Single channel duplicated) |
| **Blocks** | 3 Blocks: [Conv2D -> BatchNorm -> ReLU -> MaxPool] | Depthwise Separable Convolutions |
| **Training Strategy** | End-to-end training with class weights | Phase 1: Train Head; Phase 2: Fine-tune top layers ($1\text{e-}5$ LR) |
| **Global Pooling** | GlobalAveragePooling2D | GlobalAveragePooling2D |

---

## ⚕️ Medical Disclaimer

> **This is a research prototype developed for academic and educational purposes as a CS final-year project. It is not intended for clinical use or to serve as a substitute for professional medical diagnosis, advice, or treatment.**
