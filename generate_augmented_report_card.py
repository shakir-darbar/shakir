"""
generate_augmented_report_card.py -- Generate Terminal-Style Performance Report Cards for Phase 2

Generates dark-theme terminal-style performance calculation report cards
for the models trained on the AUGMENTED dataset.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join('results', 'augmented')
OUTPUT_DIR = os.path.join('plots', 'terminal_cards_augmented')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAMES = ['resnet50', 'densenet121', 'mobilenetv2', 'efficientnet_b0']
MODEL_DISPLAY_NAMES = {
    'resnet50': 'ResNet50 (Augmented)',
    'densenet121': 'DenseNet121 (Augmented)',
    'mobilenetv2': 'MobileNetV2 (Augmented)',
    'efficientnet_b0': 'EfficientNet-B0 (Aug)'
}

def build_card_text(model_name, results_dir=RESULTS_DIR):
    eval_json = os.path.join(results_dir, model_name, f"{model_name}_evaluation_results.json")
    report_txt = os.path.join(results_dir, model_name, "classification_report.txt")

    if not os.path.exists(eval_json) or not os.path.exists(report_txt):
        return None

    with open(eval_json, 'r') as f:
        res = json.load(f)

    with open(report_txt, 'r') as f:
        rep = f.read().strip()

    acc = res.get('accuracy', 0) * 100
    prec = res.get('macro_precision', 0) * 100
    rec = res.get('macro_recall', 0) * 100
    f1 = res.get('macro_f1', 0) * 100
    auc_val = res.get('roc_auc_macro', 0) or 0

    header = (
        f"=== Phase 2: Augmented Training Results ===\n"
        f"Accuracy  : {acc:.2f}%\n"
        f"Precision : {prec:.2f}%\n"
        f"Recall    : {rec:.2f}%\n"
        f"F1 Score  : {f1:.2f}%\n"
        f"ROC-AUC   : {auc_val:.4f}\n\n"
        f"{'-'*54}\n"
        f"CLASSIFICATION REPORT\n"
        f"{'-'*54}\n"
        f"{rep}"
    )
    return header

def render_single_card(model_name, text, output_path):
    fig, ax = plt.subplots(figsize=(7.5, 6.0), facecolor='#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    ax.axis('off')
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name.upper())

    fig.text(0.5, 0.94, display_name, color='#ffffff', fontsize=15,
             fontweight='bold', ha='center', fontfamily='sans-serif')
    rect = plt.Rectangle((0.03, 0.04), 0.94, 0.86,
                         transform=fig.transFigure,
                         facecolor='#141414', edgecolor='#3a3a3a',
                         linewidth=1.5, zorder=1)
    fig.patches.append(rect)
    fig.text(0.07, 0.86, text,
             color='#d4d4d4', fontsize=9.5, fontfamily='monospace',
             va='top', ha='left', linespacing=1.35, zorder=2)

    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

def render_combined_panel(cards_data, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#ffffff')
    fig.suptitle("Phase 2: Augmented Model Performance Summary",
                 fontsize=18, fontweight='bold', y=0.98)
    models_order = ['resnet50', 'densenet121', 'mobilenetv2', 'efficientnet_b0']

    for idx, model_key in enumerate(models_order):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        ax.set_facecolor('#1e1e1e')
        ax.axis('off')

        display_name = MODEL_DISPLAY_NAMES.get(model_key, model_key.upper())
        text = cards_data.get(model_key, "Evaluation results pending...")

        ax.text(0.5, 0.94, display_name, color='#61afef', fontsize=14,
                fontweight='bold', ha='center', va='top', transform=ax.transAxes)
        ax.text(0.06, 0.86, text,
                color='#e0e0e0', fontsize=8.5, fontfamily='monospace',
                va='top', ha='left', linespacing=1.3, transform=ax.transAxes)

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    plt.savefig(output_path, dpi=300, facecolor='#ffffff', bbox_inches='tight')
    plt.close()

def main():
    cards_data = {}
    for m in MODEL_NAMES:
        card_text = build_card_text(m)
        if card_text:
            cards_data[m] = card_text
            single_path = os.path.join(OUTPUT_DIR, f"{m}_augmented_report.png")
            render_single_card(m, card_text, single_path)
            
    if len(cards_data) == len(MODEL_NAMES):
        panel_path = os.path.join(OUTPUT_DIR, "AUGMENTED_MODELS_REPORT_PANEL.png")
        render_combined_panel(cards_data, panel_path)

if __name__ == '__main__':
    main()
