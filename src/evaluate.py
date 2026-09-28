"""
evaluate.py - Evaluation, Visualization, Model Comparison & Export Module

Performs comprehensive performance evaluation:
1. Calculates overall Accuracy, per-class Precision, Recall, and F1-Score.
2. Plots and saves Confusion Matrices (Seaborn Heatmaps).
3. Plots Loss and Accuracy training curves (Train vs Validation).
4. Generates a formal Comparison Table (Accuracy, Macro F1, Training Time, Parameter Count).
5. Compares Custom CNN vs MobileNetV2 and exports the winner as `best_model.h5`.
"""

import os
if 'KERAS_BACKEND' not in os.environ:
    os.environ['KERAS_BACKEND'] = 'tensorflow'

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import keras

from src.data_loader import TARGET_CLASSES

def plot_confusion_matrix(y_true, y_pred, class_names, title, output_path):
    """
    Generates and saves a publication-quality Confusion Matrix heatmap using Seaborn.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar=True, square=True
    )
    plt.title(title, fontsize=14, fontweight='bold', pad=12)
    plt.ylabel('True Class Label', fontsize=12)
    plt.xlabel('Predicted Class Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix plot: {output_path}")

def plot_training_curves(history_dict, model_name, output_path):
    """
    Plots training vs validation accuracy and loss learning curves.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = range(1, len(history_dict['loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    ax1.plot(epochs, history_dict['loss'], 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history_dict['val_loss'], 'r--s', label='Validation Loss', linewidth=2)
    ax1.set_title(f'{model_name} - Loss Curve', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=11)
    ax1.set_ylabel('Categorical Crossentropy Loss', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Accuracy Curve
    ax2.plot(epochs, history_dict['accuracy'], 'b-o', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history_dict['val_accuracy'], 'r--s', label='Validation Accuracy', linewidth=2)
    ax2.set_title(f'{model_name} - Accuracy Curve', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Training Curves plot: {output_path}")

def evaluate_single_model(model, X_test, y_test, model_name, class_names=TARGET_CLASSES, output_dir="plots"):
    """
    Evaluates a single model on test set, prints metrics report, and returns summary stats.
    """
    os.makedirs(output_dir, exist_ok=True)

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    labels_idx = list(range(len(class_names)))

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels_idx, average='macro', zero_division=0
    )

    print("\n" + "="*65)
    print(f"            PERFORMANCE EVALUATION REPORT: {model_name.upper()}")
    print("="*65)
    print(f"Overall Accuracy : {acc * 100:.2f}%")
    print(f"Macro Precision  : {precision * 100:.2f}%")
    print(f"Macro Recall     : {recall * 100:.2f}%")
    print(f"Macro F1-Score   : {f1 * 100:.2f}%")
    print("-" * 65)
    print("\nDetailed Per-Class Classification Report:\n")
    print(classification_report(
        y_test, y_pred, labels=labels_idx, target_names=class_names, digits=4, zero_division=0
    ))
    print("="*65 + "\n")

    # Plot Confusion Matrix
    cm_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred, class_names, f"{model_name} Confusion Matrix", cm_path)

    param_count = model.count_params()

    metrics_summary = {
        'model_name': model_name,
        'accuracy': acc,
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1,
        'param_count': param_count,
        'y_pred_probs': y_pred_probs,
        'y_pred': y_pred
    }

    return metrics_summary

def generate_comparison_table(results_list):
    """
    Generates and prints a clean comparative summary table for Custom CNN vs MobileNetV2.
    """
    table_data = []
    for res in results_list:
        table_data.append({
            'Model Architecture': res['model_name'],
            'Accuracy (%)': f"{res['accuracy'] * 100:.2f}%",
            'Macro F1-Score': f"{res['macro_f1']:.4f}",
            'Training Time (s)': f"{res['train_time']:.2f}s",
            'Total Parameters': f"{res['param_count']:,}"
        })

    df_comp = pd.DataFrame(table_data)

    print("\n" + "="*75)
    print("         MODEL COMPARISON SUMMARY (CUSTOM CNN vs MOBILENETV2)")
    print("="*75)
    print(df_comp.to_string(index=False))
    print("="*75 + "\n")

    return df_comp

def save_best_model(model_a, res_a, model_b, res_b, export_dir="."):
    """
    Compares Macro F1-Scores of both models, selects the champion model,
    saves it to `best_model.h5`, and updates `label_mapping.json`.
    """
    os.makedirs(export_dir, exist_ok=True)
    best_model_path = os.path.join(export_dir, "best_model.h5")
    mapping_path = os.path.join(export_dir, "label_mapping.json")

    if res_a['macro_f1'] >= res_b['macro_f1']:
        champion_model = model_a
        winner_name = res_a['model_name']
        winner_f1 = res_a['macro_f1']
    else:
        champion_model = model_b
        winner_name = res_b['model_name']
        winner_f1 = res_b['macro_f1']

    print("\n" + "*"*65)
    print(f" CHAMPION MODEL SELECTION: {winner_name} (Macro F1: {winner_f1:.4f})")
    print(f" Saving champion model to: {os.path.abspath(best_model_path)}")
    print("*"*65 + "\n")

    # Save model weights & architecture
    champion_model.save(best_model_path)

    # Save mapping json
    mapping_data = {
        'winning_model': winner_name,
        'macro_f1': float(winner_f1),
        'target_classes': TARGET_CLASSES,
        'class_to_idx': {cls: i for i, cls in enumerate(TARGET_CLASSES)},
        'idx_to_class': {i: cls for i, cls in enumerate(TARGET_CLASSES)}
    }
    with open(mapping_path, 'w') as f:
        json.dump(mapping_data, f, indent=4)

    return winner_name, best_model_path
