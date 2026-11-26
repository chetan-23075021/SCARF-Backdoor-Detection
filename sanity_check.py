"""
Pre-flight sanity checks for the SCARF pipeline.
Run this BEFORE any expensive training so you don't burn Colab time
on a run that's invalidated by a wrong assumption.

Usage:
    python sanity_check.py
"""

import numpy as np
import os
from config import config

def check_person2_zero():
    """
    The completion-model math (train_completion.py's data.mean(dim=-1) target,
    and models.py's recon * 2.0 correction) is only valid if person-slot M=1
    is all-zero padding for every sample. Standard NTU-60 (60 classes) has
    ~11 two-person interaction classes where this is FALSE.
    """
    train_path = os.path.join(config.DATA_PATH, 'train_data.npy')
    print(f"Loading (mmap, read-only, fast) from: {train_path}")
    data = np.load(train_path, mmap_mode='r')  # mmap = no full read, fast
    print(f"  Shape: {data.shape}  (N, C, T, V, M)")

    n_check = min(2000, data.shape[0])
    idx = np.random.choice(data.shape[0], n_check, replace=False)

    max_abs_person2 = 0.0
    n_nonzero_person2 = 0
    for i in idx:
        sample = data[i]                       # (C, T, V, M)
        p2 = sample[..., 1]                    # second person slot
        m = np.abs(p2).max()
        if m > 1e-6:
            n_nonzero_person2 += 1
        max_abs_person2 = max(max_abs_person2, m)

    print(f"\n  Checked {n_check} random samples.")
    print(f"  Samples where person-2 is NOT all-zero: {n_nonzero_person2} "
          f"({100*n_nonzero_person2/n_check:.1f}%)")
    print(f"  Max |value| seen in person-2 slot: {max_abs_person2:.4f}")

    if n_nonzero_person2 > 0:
        print("\n  ⚠️  WARNING: person-2 is populated in some samples.")
        print("      The 'recon * 2.0' fix in models.py and the mean(dim=-1)")
        print("      training target in train_completion.py assume person-2")
        print("      is always zero. For the flagged samples that assumption")
        print("      is FALSE -- expect noisier completion/detection results")
        print("      specifically on two-person interaction classes.")
        print("      Consider: (a) filtering to single-actor classes only, or")
        print("      (b) training/running the completion model on actor-0")
        print("      only everywhere, dropping the mean/×2 approach entirely.")
    else:
        print("\n  ✓ Person-2 is all-zero in every sample checked. The ×2.0")
        print("    correction and mean(dim=-1) training target are valid.")


def check_config_vs_checkpoints():
    """
    Layer shapes (fc, embed) are baked into saved state_dicts. A config
    change to NUM_CLASSES/NUM_JOINTS/INPUT_CHANNELS after a checkpoint was
    saved will silently corrupt (or crash) a resumed/loaded run.
    """
    print(f"\nCurrent config: NUM_CLASSES={config.NUM_CLASSES}, "
          f"NUM_JOINTS={config.NUM_JOINTS}, INPUT_CHANNELS={config.INPUT_CHANNELS}")
    ckpt_dir = config.CHECKPOINT_DIR
    if not os.path.isdir(ckpt_dir):
        print("  No checkpoint directory yet -- nothing to check.")
        return

    import torch
    for fname in os.listdir(ckpt_dir):
        if not fname.endswith('.pth'):
            continue
        path = os.path.join(ckpt_dir, fname)
        try:
            state = torch.load(path, map_location='cpu')
            sd = state.get('model_state_dict', state)  # handle both checkpoint formats
            if 'fc.weight' in sd:
                ckpt_classes = sd['fc.weight'].shape[0]
                match = "✓" if ckpt_classes == config.NUM_CLASSES else "✗ MISMATCH"
                print(f"  {fname}: fc outputs {ckpt_classes} classes  [{match}]")
        except Exception as e:
            print(f"  {fname}: could not inspect ({e})")


def check_labels_in_range():
    """Confirm labels actually fall within [0, NUM_CLASSES)."""
    label_path = os.path.join(config.DATA_PATH, 'train_label.pkl')
    import pickle
    with open(label_path, 'rb') as f:
        loaded = pickle.load(f)
    labels = loaded[1] if isinstance(loaded, tuple) and len(loaded) == 2 else loaded
    labels = np.array(labels)
    print(f"\nLabel range: min={labels.min()}, max={labels.max()}, "
          f"unique={len(np.unique(labels))}, config.NUM_CLASSES={config.NUM_CLASSES}")
    if labels.max() >= config.NUM_CLASSES:
        print("  ✗ MISMATCH: labels exceed NUM_CLASSES -- model output layer too small.")
    else:
        print("  ✓ Labels fit within configured NUM_CLASSES.")


if __name__ == "__main__":
    print("=" * 60)
    print("SCARF pre-flight sanity checks")
    print("=" * 60)
    check_labels_in_range()
    print()
    check_config_vs_checkpoints()
    print()
    check_person2_zero()
    print("\n" + "=" * 60)
    print("Done. Fix any ✗/⚠️ above before running the full pipeline.")
    print("=" * 60)