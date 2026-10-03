"""
evaluate.py -- Comprehensive Model Evaluation & Visualization Module

Generates all required evaluation outputs for each CNN model:
    1.  Accuracy
    2.  Precision (per-class, macro, weighted)
    3.  Recall (per-class, macro, weighted)
    4.  F1-score (per-class, macro, weighted)
    5.  ROC-AUC (per-class, macro)
    6.  Per-class ROC curves
    7.  Confusion matrix heatmap
    8.  Classification report
    9.  Training accuracy/loss curves
    10. All results saved to disk (JSON + PNG plots)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

from src.data_pipeline import TARGET_CLASSES, NUM_CLASSES


def evaluate_model(model, test_dataset, model_name, output_dir, history_dict=None,
                   train_time=None, model_info=None):
    """
    Performs comprehensive evaluation of a trained model on the test set.

    Parameters:
        model: Trained Keras model.
        test_dataset: tf.data.Dataset for the test set.
        model_name (str): Name of the model (e.g., 'resnet50').
        output_dir (str): Directory to save evaluation outputs.
        history_dict (dict): Training history for plotting curves.
        train_time (float): Training time in seconds.
        model_info (dict): Parameter count info.

    Returns:
        dict: Complete evaluation results.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  EVALUATING: {model_name.upper()} ON TEST SET")
    print("=" * 70)

    # ── 1. Get predictions ──────────────────────────────────────────────
    print("\n[1/6] Generating predictions on test set...")

    y_true = []
    y_pred_probs = []

    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0)
        y_pred_probs.append(preds)
        y_true.append(labels.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    num_test_samples = len(y_true)
    print(f"  Test samples: {num_test_samples}")

    # ── 2. Compute metrics ──────────────────────────────────────────────
    print("[2/6] Computing classification metrics...")

    acc = accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), average=None, zero_division=0
    )

    # Macro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), average='macro', zero_division=0
    )

    # Weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), average='weighted', zero_division=0
    )

    # ROC-AUC (One-vs-Rest)
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    # Per-class AUC
    per_class_auc = {}
    for i, cls_name in enumerate(TARGET_CLASSES):
        if len(np.unique(y_true_bin[:, i])) > 1:  # Need both classes present
            per_class_auc[cls_name] = float(roc_auc_score(y_true_bin[:, i], y_pred_probs[:, i]))
        else:
            per_class_auc[cls_name] = float('nan')

    # Macro-average AUC
    try:
        macro_auc = roc_auc_score(y_true_bin, y_pred_probs, average='macro', multi_class='ovr')
    except ValueError:
        macro_auc = float('nan')

    # ── 3. Print results ────────────────────────────────────────────────
    print("\n" + "-" * 65)
    print(f"  {model_name.upper()} -- TEST SET RESULTS")
    print("-" * 65)
    print(f"  Accuracy          : {acc*100:.2f}%")
    print(f"  Macro Precision   : {macro_precision*100:.2f}%")
    print(f"  Macro Recall      : {macro_recall*100:.2f}%")
    print(f"  Macro F1-Score    : {macro_f1*100:.2f}%")
    print(f"  Weighted Precision: {weighted_precision*100:.2f}%")
    print(f"  Weighted Recall   : {weighted_recall*100:.2f}%")
    print(f"  Weighted F1-Score : {weighted_f1*100:.2f}%")
    print(f"  ROC-AUC (Macro)   : {macro_auc:.4f}")
    print("-" * 65)

    print("\n  Per-class metrics:")
    print(f"  {'Class':<18} | {'Prec':>7} | {'Recall':>7} | {'F1':>7} | {'AUC':>7} | {'Support':>7}")
    print(f"  {'-'*18}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    for i, cls in enumerate(TARGET_CLASSES):
        cls_auc = per_class_auc.get(cls, float('nan'))
        auc_str = f"{cls_auc:.4f}" if not np.isnan(cls_auc) else "N/A"
        print(f"  {cls:<18} | {precision_per[i]*100:>6.2f}% | {recall_per[i]*100:>6.2f}% | "
              f"{f1_per[i]*100:>6.2f}% | {auc_str:>7} | {int(support_per[i]):>7}")

    report_text = classification_report(y_true, y_pred, labels=list(range(NUM_CLASSES)),
                                         target_names=TARGET_CLASSES, digits=4, zero_division=0)
    print("\n  Full Classification Report:")
    print(report_text)

    # Save classification report to disk
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)
    with open(os.path.join(output_dir, f"{model_name}_classification_report.txt"), 'w') as f:
        f.write(report_text)
    print(f"  Saved: {report_path}")

    # ── 4. Plot confusion matrix ────────────────────────────────────────
    print("[3/6] Generating confusion matrices (raw + normalized)...")
    _plot_confusion_matrix(y_true, y_pred, TARGET_CLASSES, model_name, output_dir)

    # ── 5. Plot ROC curves ──────────────────────────────────────────────
    print("[4/6] Generating ROC curves...")
    _plot_roc_curves(y_true_bin, y_pred_probs, TARGET_CLASSES, model_name, output_dir)

    # ── 6. Plot training curves ─────────────────────────────────────────
    if history_dict:
        print("[5/6] Generating training curves (loss & accuracy)...")
        _plot_training_curves(history_dict, model_name, output_dir)
    else:
        print("[5/6] No training history provided, skipping training curves.")

    # ── 7. Save results to JSON ─────────────────────────────────────────
    print("[6/6] Saving evaluation results...")

    results = {
        'model_name': model_name,
        'test_samples': int(num_test_samples),
        'accuracy': float(acc),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_precision),
        'weighted_recall': float(weighted_recall),
        'weighted_f1': float(weighted_f1),
        'roc_auc_macro': float(macro_auc) if not np.isnan(macro_auc) else None,
        'per_class': {},
        'training_time_seconds': float(train_time) if train_time else None,
        'total_params': model_info['total_params'] if model_info else None,
        'trainable_params': model_info['trainable_params'] if model_info else None,
    }

    for i, cls in enumerate(TARGET_CLASSES):
        results['per_class'][cls] = {
            'precision': float(precision_per[i]),
            'recall': float(recall_per[i]),
            'f1': float(f1_per[i]),
            'support': int(support_per[i]),
            'auc': float(per_class_auc.get(cls, float('nan')))
        }

    results_path = os.path.join(output_dir, f"{model_name}_evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"  Results saved to: {results_path}")

    print("\n" + "=" * 70)
    print(f"  {model_name.upper()} EVALUATION COMPLETE")
    print("=" * 70 + "\n")

    return results


def _plot_confusion_matrix(y_true, y_pred, class_names, model_name, output_dir):
    """Generates and saves raw and normalized confusion matrix heatmaps."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    # 1. Raw counts
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar=True, square=True, ax=ax,
        annot_kws={'size': 14}
    )
    ax.set_title(f'{model_name} -- Confusion Matrix', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"), dpi=300)
    plt.close()
    print(f"  Saved: {cm_path}")

    # 2. Normalized
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar=True, square=True, ax=ax,
        annot_kws={'size': 13}
    )
    ax.set_title(f'{model_name} -- Normalized Confusion Matrix', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    norm_path = os.path.join(output_dir, "normalized_confusion_matrix.png")
    plt.savefig(norm_path, dpi=300)
    plt.savefig(os.path.join(output_dir, f"{model_name}_normalized_confusion_matrix.png"), dpi=300)
    plt.close()
    print(f"  Saved: {norm_path}")


def _plot_roc_curves(y_true_bin, y_pred_probs, class_names, model_name, output_dir):
    """Generates per-class ROC curves and macro-average ROC."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    # Per-class ROC
    all_fpr = []
    all_tpr = []
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        if len(np.unique(y_true_bin[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            roc_auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f'{cls} (AUC = {roc_auc_val:.4f})')
            all_fpr.append(fpr)
            all_tpr.append(tpr)

    # Macro-average ROC
    if all_fpr:
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(mean_fpr)
        for fpr_i, tpr_i in zip(all_fpr, all_tpr):
            mean_tpr += np.interp(mean_fpr, fpr_i, tpr_i)
        mean_tpr /= len(all_fpr)
        mean_auc = auc(mean_fpr, mean_tpr)
        ax.plot(mean_fpr, mean_tpr, color='black', lw=2.5, linestyle='--',
                label=f'Macro Average (AUC = {mean_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k:', lw=1, alpha=0.5, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{model_name} -- ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    roc_path = os.path.join(output_dir, f"roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.savefig(os.path.join(output_dir, f"{model_name}_roc_curves.png"), dpi=300)
    plt.close()
    print(f"  Saved: {roc_path}")


def _plot_training_curves(history_dict, model_name, output_dir):
    """Plots training vs validation accuracy and loss curves."""
    epochs = range(1, len(history_dict['loss']) + 1)

    # 1. Combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history_dict['loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, history_dict['val_loss'], 'r--s', label='Val Loss', linewidth=2, markersize=4)
    ax1.set_title(f'{model_name} -- Loss Curve', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(epochs, history_dict['accuracy'], 'b-o', label='Train Accuracy', linewidth=2, markersize=4)
    ax2.plot(epochs, history_dict['val_accuracy'], 'r--s', label='Val Accuracy', linewidth=2, markersize=4)
    ax2.set_title(f'{model_name} -- Accuracy Curve', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    curves_path = os.path.join(output_dir, f"{model_name}_training_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()

    # 2. Individual loss_curve.png
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history_dict['loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax.plot(epochs, history_dict['val_loss'], 'r--s', label='Val Loss', linewidth=2, markersize=4)
    ax.set_title(f'{model_name} -- Training & Validation Loss', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=300)
    plt.close()

    # 3. Individual accuracy_curve.png
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history_dict['accuracy'], 'b-o', label='Train Accuracy', linewidth=2, markersize=4)
    ax.plot(epochs, history_dict['val_accuracy'], 'r--s', label='Val Accuracy', linewidth=2, markersize=4)
    ax.set_title(f'{model_name} -- Training & Validation Accuracy', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"), dpi=300)
    plt.close()

    print(f"  Saved: loss_curve.png, accuracy_curve.png, {curves_path}")


def generate_comparison_table(results_list, output_path=None):
    """
    Generates and prints a formatted comparison table across all models.
    Saves results to JSON, CSV, and formatted table.
    """
    import pandas as pd

    print("\n" + "=" * 95)
    print("  MODEL COMPARISON TABLE")
    print("=" * 95)

    header = (f"  {'Model':<18} | {'Acc':>7} | {'Prec':>7} | {'Recall':>7} | "
              f"{'MacroF1':>7} | {'WtdF1':>7} | {'AUC':>7} | {'Params':>12} | {'Time':>8}")
    print(header)
    print(f"  {'-'*18}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*12}-+-{'-'*8}")

    rows = []
    for r in results_list:
        name = r['model_name']
        acc = r['accuracy'] * 100
        prec = r['macro_precision'] * 100
        rec = r['macro_recall'] * 100
        mf1 = r['macro_f1'] * 100
        wf1 = r['weighted_f1'] * 100
        auc_val = r.get('roc_auc_macro', 0) or 0
        params = r.get('total_params', 0) or 0
        trainable = r.get('trainable_params', 0) or 0
        time_s = r.get('training_time_seconds', 0) or 0

        print(f"  {name:<18} | {acc:>6.2f}% | {prec:>6.2f}% | {rec:>6.2f}% | "
              f"{mf1:>6.2f}% | {wf1:>6.2f}% | {auc_val:>7.4f} | {params:>12,} | {time_s:>7.1f}s")

        rows.append({
            'Model': name,
            'Accuracy': round(acc, 2),
            'Macro Precision': round(prec, 2),
            'Weighted Precision': round(r['weighted_precision'] * 100, 2),
            'Macro Recall': round(rec, 2),
            'Weighted Recall': round(r['weighted_recall'] * 100, 2),
            'Macro F1': round(mf1, 2),
            'Weighted F1': round(wf1, 2),
            'Macro ROC-AUC': round(auc_val, 4),
            'Total Parameters': params,
            'Trainable Parameters': trainable,
            'Training Time (s)': round(time_s, 1)
        })

    print("=" * 95 + "\n")

    if output_path:
        out_dir = os.path.dirname(output_path)
        os.makedirs(out_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results_list, f, indent=4)
        print(f"  Comparison JSON saved to: {output_path}")

        # Also save CSV
        df = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, "CNN_MODEL_COMPARISON.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Comparison CSV saved to : {csv_path}\n")

