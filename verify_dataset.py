"""
verify_dataset.py -- PNG Dataset Verification Script

Performs comprehensive integrity checks on the generated PNG spectrogram dataset:
  1. Directory structure validation
  2. Class label verification
  3. Image count per class per split
  4. Image integrity (open & validate every PNG)
  5. Image dimension consistency
  6. Duplicate detection (MD5 hashing)
  7. Patient leakage re-verification
  8. Distribution analysis
"""

import os
import sys
import json
import hashlib
from collections import defaultdict
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ── Configuration ────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join("data", "original_png")
EXPECTED_SPLITS = ["train", "validation", "test"]
EXPECTED_CLASSES = ["Healthy", "Pneumonia", "URTI", "Bronchiectasis"]
EXPECTED_WIDTH = 157
EXPECTED_HEIGHT = 128
EXPECTED_MODE = "RGB"


def md5_file(filepath):
    """Compute MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_patient_id_from_png(filename):
    """
    Extracts patient ID from PNG filename.
    Format: {patientID}_{recordingInfo}_seg{N}.png
    Example: 101_1b1_Al_sc_Medtron_seg00.png -> 101
    """
    parts = filename.split('_')
    if parts and parts[0].isdigit():
        return int(parts[0])
    return None


def main():
    print("\n" + "=" * 70)
    print("  STEP 3: DATASET VERIFICATION REPORT")
    print("=" * 70)

    all_passed = True
    warnings = []

    # ── CHECK 1: Directory Structure ─────────────────────────────────────
    print("\n[CHECK 1] Directory Structure Validation")
    print("-" * 50)

    if not os.path.exists(DATASET_DIR):
        print(f"  [FAIL] Dataset directory not found: {DATASET_DIR}")
        return

    missing_dirs = []
    for split in EXPECTED_SPLITS:
        split_dir = os.path.join(DATASET_DIR, split)
        if not os.path.exists(split_dir):
            missing_dirs.append(split_dir)
            continue
        for cls in EXPECTED_CLASSES:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.exists(cls_dir):
                missing_dirs.append(cls_dir)

    if missing_dirs:
        print(f"  [FAIL] Missing directories:")
        for d in missing_dirs:
            print(f"    - {d}")
        all_passed = False
    else:
        print(f"  [PASS] All {len(EXPECTED_SPLITS)} splits x {len(EXPECTED_CLASSES)} classes = "
              f"{len(EXPECTED_SPLITS) * len(EXPECTED_CLASSES)} directories found.")

    # ── CHECK 2: Class Label Verification ────────────────────────────────
    print("\n[CHECK 2] Class Label Verification")
    print("-" * 50)

    unexpected_classes = []
    for split in EXPECTED_SPLITS:
        split_dir = os.path.join(DATASET_DIR, split)
        if os.path.exists(split_dir):
            actual_classes = [d for d in os.listdir(split_dir)
                              if os.path.isdir(os.path.join(split_dir, d))]
            for cls in actual_classes:
                if cls not in EXPECTED_CLASSES:
                    unexpected_classes.append(f"{split}/{cls}")

    if unexpected_classes:
        print(f"  [FAIL] Unexpected class folders found:")
        for uc in unexpected_classes:
            print(f"    - {uc}")
        all_passed = False
    else:
        print(f"  [PASS] Only expected classes found: {EXPECTED_CLASSES}")

    # ── CHECK 3: Image Count Per Class Per Split ─────────────────────────
    print("\n[CHECK 3] Image Count Per Class Per Split")
    print("-" * 50)

    counts = {}
    total_files = 0
    non_png_files = []

    for split in EXPECTED_SPLITS:
        counts[split] = {}
        for cls in EXPECTED_CLASSES:
            cls_dir = os.path.join(DATASET_DIR, split, cls)
            if os.path.exists(cls_dir):
                all_files = os.listdir(cls_dir)
                png_files = [f for f in all_files if f.lower().endswith('.png')]
                non_png = [f for f in all_files if not f.lower().endswith('.png')]
                counts[split][cls] = len(png_files)
                total_files += len(png_files)
                if non_png:
                    non_png_files.extend([os.path.join(split, cls, f) for f in non_png])
            else:
                counts[split][cls] = 0

    print(f"\n  {'Class':<18} | {'Train':>7} | {'Val':>7} | {'Test':>7} | {'Total':>7}")
    print(f"  {'-'*18}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    for cls in EXPECTED_CLASSES:
        train_c = counts.get('train', {}).get(cls, 0)
        val_c = counts.get('validation', {}).get(cls, 0)
        test_c = counts.get('test', {}).get(cls, 0)
        total_c = train_c + val_c + test_c
        print(f"  {cls:<18} | {train_c:>7} | {val_c:>7} | {test_c:>7} | {total_c:>7}")

    grand_train = sum(counts.get('train', {}).values())
    grand_val = sum(counts.get('validation', {}).values())
    grand_test = sum(counts.get('test', {}).values())
    print(f"  {'TOTAL':<18} | {grand_train:>7} | {grand_val:>7} | {grand_test:>7} | {total_files:>7}")

    if non_png_files:
        print(f"\n  [WARNING] Non-PNG files found ({len(non_png_files)}):")
        for f in non_png_files[:5]:
            print(f"    - {f}")
        warnings.append(f"{len(non_png_files)} non-PNG files found in class directories")

    # Check for empty classes
    empty_classes = []
    for split in EXPECTED_SPLITS:
        for cls in EXPECTED_CLASSES:
            if counts.get(split, {}).get(cls, 0) == 0:
                empty_classes.append(f"{split}/{cls}")

    if empty_classes:
        print(f"\n  [FAIL] Empty class directories:")
        for ec in empty_classes:
            print(f"    - {ec}")
        all_passed = False
    else:
        print(f"\n  [PASS] No empty class directories. Total: {total_files} PNG files.")

    # ── CHECK 4: Image Integrity (Open & Validate Every PNG) ─────────────
    print("\n[CHECK 4] Image Integrity Verification")
    print("-" * 50)

    corrupted = []
    checked = 0

    for split in EXPECTED_SPLITS:
        for cls in EXPECTED_CLASSES:
            cls_dir = os.path.join(DATASET_DIR, split, cls)
            if not os.path.exists(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith('.png'):
                    continue
                fpath = os.path.join(cls_dir, fname)
                try:
                    img = Image.open(fpath)
                    img.verify()  # Verify image integrity
                    checked += 1
                except Exception as e:
                    corrupted.append((fpath, str(e)))

    if corrupted:
        print(f"  [FAIL] {len(corrupted)} corrupted images found:")
        for fp, err in corrupted[:10]:
            print(f"    - {fp}: {err}")
        all_passed = False
    else:
        print(f"  [PASS] All {checked} PNG files opened and verified successfully.")

    # ── CHECK 5: Image Dimension Consistency ─────────────────────────────
    print("\n[CHECK 5] Image Dimension Consistency")
    print("-" * 50)

    wrong_dims = []
    wrong_mode = []
    checked_dims = 0

    for split in EXPECTED_SPLITS:
        for cls in EXPECTED_CLASSES:
            cls_dir = os.path.join(DATASET_DIR, split, cls)
            if not os.path.exists(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith('.png'):
                    continue
                fpath = os.path.join(cls_dir, fname)
                try:
                    img = Image.open(fpath)
                    w, h = img.size
                    if w != EXPECTED_WIDTH or h != EXPECTED_HEIGHT:
                        wrong_dims.append((fpath, w, h))
                    if img.mode != EXPECTED_MODE:
                        wrong_mode.append((fpath, img.mode))
                    checked_dims += 1
                except Exception:
                    pass  # Already caught in CHECK 4

    if wrong_dims:
        print(f"  [FAIL] {len(wrong_dims)} images with unexpected dimensions:")
        for fp, w, h in wrong_dims[:5]:
            print(f"    - {fp}: {w}x{h} (expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT})")
        all_passed = False
    else:
        print(f"  [PASS] All {checked_dims} images are {EXPECTED_WIDTH}x{EXPECTED_HEIGHT} pixels.")

    if wrong_mode:
        print(f"  [FAIL] {len(wrong_mode)} images with wrong color mode:")
        for fp, mode in wrong_mode[:5]:
            print(f"    - {fp}: {mode} (expected {EXPECTED_MODE})")
        all_passed = False
    else:
        print(f"  [PASS] All {checked_dims} images are {EXPECTED_MODE} mode (3-channel).")

    # ── CHECK 6: Duplicate Detection (MD5 Hashing) ──────────────────────
    print("\n[CHECK 6] Duplicate Image Detection (MD5)")
    print("-" * 50)

    hash_map = defaultdict(list)

    for split in EXPECTED_SPLITS:
        for cls in EXPECTED_CLASSES:
            cls_dir = os.path.join(DATASET_DIR, split, cls)
            if not os.path.exists(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith('.png'):
                    continue
                fpath = os.path.join(cls_dir, fname)
                file_hash = md5_file(fpath)
                hash_map[file_hash].append(os.path.join(split, cls, fname))

    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}

    if duplicates:
        total_dupes = sum(len(p) - 1 for p in duplicates.values())
        print(f"  [WARNING] {total_dupes} duplicate images found ({len(duplicates)} groups):")
        shown = 0
        for h, paths in list(duplicates.items())[:5]:
            print(f"    Hash {h[:12]}...:")
            for p in paths:
                print(f"      - {p}")
            shown += 1
        warnings.append(f"{total_dupes} duplicate images detected")
    else:
        print(f"  [PASS] No duplicate images found. All {len(hash_map)} files are unique.")

    # ── CHECK 7: Patient Leakage Re-Verification ────────────────────────
    print("\n[CHECK 7] Patient Leakage Re-Verification")
    print("-" * 50)

    split_patients = defaultdict(set)

    for split in EXPECTED_SPLITS:
        for cls in EXPECTED_CLASSES:
            cls_dir = os.path.join(DATASET_DIR, split, cls)
            if not os.path.exists(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith('.png'):
                    continue
                pid = extract_patient_id_from_png(fname)
                if pid is not None:
                    split_patients[split].add(pid)

    train_pids = split_patients.get('train', set())
    val_pids = split_patients.get('validation', set())
    test_pids = split_patients.get('test', set())

    overlap_tv = train_pids & val_pids
    overlap_tt = train_pids & test_pids
    overlap_vt = val_pids & test_pids

    print(f"  Train patients      : {len(train_pids)}")
    print(f"  Validation patients : {len(val_pids)}")
    print(f"  Test patients       : {len(test_pids)}")
    print(f"  Train & Val overlap : {len(overlap_tv)}")
    print(f"  Train & Test overlap: {len(overlap_tt)}")
    print(f"  Val & Test overlap  : {len(overlap_vt)}")

    if overlap_tv or overlap_tt or overlap_vt:
        print(f"  [FAIL] PATIENT LEAKAGE DETECTED!")
        if overlap_tv:
            print(f"    Train & Val shared patients: {sorted(overlap_tv)}")
        if overlap_tt:
            print(f"    Train & Test shared patients: {sorted(overlap_tt)}")
        if overlap_vt:
            print(f"    Val & Test shared patients: {sorted(overlap_vt)}")
        all_passed = False
    else:
        print(f"  [PASS] Zero patient overlap across all splits. No leakage.")

    # ── CHECK 8: Distribution Analysis ───────────────────────────────────
    print("\n[CHECK 8] Distribution Analysis")
    print("-" * 50)

    # Per-split percentages
    for split in EXPECTED_SPLITS:
        split_total = sum(counts.get(split, {}).values())
        if split_total == 0:
            continue
        print(f"\n  {split.upper()} distribution:")
        for cls in EXPECTED_CLASSES:
            c = counts.get(split, {}).get(cls, 0)
            pct = (c / split_total * 100) if split_total > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"    {cls:<18}: {c:>4} ({pct:5.1f}%) {bar}")

    # Overall split ratio
    print(f"\n  Overall split ratio:")
    print(f"    Train      : {grand_train:>5} ({grand_train/total_files*100:.1f}%)")
    print(f"    Validation : {grand_val:>5} ({grand_val/total_files*100:.1f}%)")
    print(f"    Test       : {grand_test:>5} ({grand_test/total_files*100:.1f}%)")

    # Class imbalance ratio (max/min per split)
    print(f"\n  Class imbalance ratio (max_count / min_count per split):")
    for split in EXPECTED_SPLITS:
        split_counts = [counts.get(split, {}).get(cls, 0) for cls in EXPECTED_CLASSES]
        if min(split_counts) > 0:
            ratio = max(split_counts) / min(split_counts)
            print(f"    {split:<12}: {ratio:.2f}x  (max={max(split_counts)}, min={min(split_counts)})")
        else:
            print(f"    {split:<12}: N/A (empty class exists)")

    # ── FINAL VERDICT ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  VERIFICATION SUMMARY")
    print("=" * 70)

    checks = [
        ("Directory Structure", True if not missing_dirs else False),
        ("Class Labels", True if not unexpected_classes else False),
        ("Image Counts", True if not empty_classes else False),
        ("Image Integrity", True if not corrupted else False),
        ("Image Dimensions", True if not wrong_dims else False),
        ("Color Mode", True if not wrong_mode else False),
        ("No Duplicates", True if not duplicates else False),
        ("No Patient Leakage", True if not (overlap_tv or overlap_tt or overlap_vt) else False),
    ]

    for name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")

    if all_passed and not warnings:
        print(f"\n  >>> ALL CHECKS PASSED. Dataset is ready for CNN training. <<<")
    elif all_passed and warnings:
        print(f"\n  >>> ALL CHECKS PASSED with {len(warnings)} warning(s). Review warnings above. <<<")
    else:
        print(f"\n  >>> SOME CHECKS FAILED. Do NOT proceed to training until issues are resolved. <<<")

    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    main()
