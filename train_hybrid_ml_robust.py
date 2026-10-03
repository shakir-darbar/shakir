"""
train_hybrid_ml_robust.py -- Extremely Memory-Efficient ML Hybrid

This version avoids TensorFlow dataset hangs by extracting features 
image-by-image, saving them to disk, clearing RAM, and then training
the Random Forest/XGBoost classifier on the combined 3072 features.
"""

import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import gc

# Prevent TF from taking all memory
physical_devices = tf.config.list_physical_devices('GPU')
try:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

DATA_DIR = os.path.join('data', 'augmented_png')
CLASSES = ['Bronchiectasis', 'Healthy', 'Pneumonia', 'URTI']
IMAGE_SIZE = (224, 224)

def get_gap_layer_model(model_path):
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(model_path)
    for layer in reversed(model.layers):
        if 'global_average_pooling' in layer.name.lower() or 'gap' in layer.name.lower():
            return tf.keras.Model(inputs=model.input, outputs=layer.output)
    return tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

def extract_features_to_disk(model_path, model_name):
    print(f"\n--- Extracting Features for {model_name} ---")
    model = get_gap_layer_model(model_path)
    
    for split in ['train', 'validation', 'test']:
        print(f"Processing {split} set...")
        features = []
        labels = []
        
        for class_idx, class_name in enumerate(CLASSES):
            img_paths = glob.glob(os.path.join(DATA_DIR, split, class_name, '*.png'))
            for img_path in img_paths:
                # Load image manually to bypass TF dataset memory leaks
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, 0)
                
                feat = model(img_array, training=False).numpy()[0]
                features.append(feat)
                labels.append(class_idx)
                
        np.save(f"features_{model_name}_{split}_x.npy", np.array(features))
        np.save(f"features_{model_name}_{split}_y.npy", np.array(labels))
        print(f"Saved {len(features)} {split} features.")
        
    del model
    gc.collect()

def main():
    if not os.path.exists('features_dense_train_x.npy'):
        extract_features_to_disk(r'models_augmented\densenet121\densenet121_best.keras', 'dense')
        extract_features_to_disk(r'models_augmented\resnet50\resnet50_best.keras', 'resnet')
    else:
        print("Features already found on disk! Skipping extraction.")
    
    print("\n--- Training ML Classifier ---")
    
    # Load and combine DenseNet + ResNet features
    print("Combining 1024 DenseNet features with 2048 ResNet features...")
    X_train = np.concatenate([np.load('features_dense_train_x.npy'), np.load('features_resnet_train_x.npy')], axis=1)
    y_train = np.load('features_dense_train_y.npy')
    
    X_val = np.concatenate([np.load('features_dense_validation_x.npy'), np.load('features_resnet_validation_x.npy')], axis=1)
    y_val = np.load('features_dense_validation_y.npy')
    
    X_test = np.concatenate([np.load('features_dense_test_x.npy'), np.load('features_resnet_test_x.npy')], axis=1)
    y_test = np.load('features_dense_test_y.npy')
    
    # Combine Train and Validation
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)
    
    print(f"Training on {X_train_full.shape[0]} samples, {X_train_full.shape[1]} total features.")
    
    try:
        import xgboost as xgb
        print("--> Using XGBoost Classifier...")
        clf = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
    except ImportError:
        print("--> XGBoost not installed. Using Random Forest Classifier...")
        clf = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1, class_weight='balanced')
        
    clf.fit(X_train_full, y_train_full)
    
    print("\nEvaluating on Unseen Test Set...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f"FINAL CNN + ML HYBRID ACCURACY: {acc*100:.2f}%")
    print("="*50)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))

if __name__ == '__main__':
    main()
