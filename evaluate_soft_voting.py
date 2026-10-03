"""
evaluate_soft_voting.py -- Soft Voting Ensemble

This script loads the predictions of our best models from Phase 2
and averages their output probabilities (Soft Voting) to get the final diagnosis.
This avoids the massive overfitting seen in deep feature fusion.
"""

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from src.data_pipeline import load_dataset

DATA_DIR = os.path.join('data', 'augmented_png')

def main():
    print("Loading Best Models from Phase 2...")
    dense_model = tf.keras.models.load_model(r'models_augmented\densenet121\densenet121_best.keras')
    resnet_model = tf.keras.models.load_model(r'models_augmented\resnet50\resnet50_best.keras')
    
    test_ds = load_dataset(DATA_DIR, split='test')
    
    y_true = []
    dense_preds = []
    resnet_preds = []
    
    print("\nGenerating Predictions on Test Set...")
    for images, labels in test_ds:
        y_true.append(labels.numpy())
        dense_preds.append(dense_model.predict(images, verbose=0))
        resnet_preds.append(resnet_model.predict(images, verbose=0))
        
    y_true = np.concatenate(y_true, axis=0)
    dense_preds = np.concatenate(dense_preds, axis=0)
    resnet_preds = np.concatenate(resnet_preds, axis=0)
    
    # Soft Voting: Average the probabilities
    ensemble_preds = (dense_preds + resnet_preds) / 2.0
    y_pred = np.argmax(ensemble_preds, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "=" * 50)
    print(f"SOFT VOTING ENSEMBLE ACCURACY: {acc*100:.2f}%")
    print("=" * 50)

if __name__ == '__main__':
    main()
