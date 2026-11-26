import os
import sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from models import STGCN
from data_loader import get_dataloader
from inject_poison import poison_dataset


def load_victim(device=None):
    device = device or torch.device(config.DEVICE)
    victim = STGCN(num_classes=config.NUM_CLASSES).to(device)
    path = os.path.join(config.CHECKPOINT_DIR, 'best_victim.pth')
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run SCARF victim training first.")
    victim.load_state_dict(torch.load(path, map_location=device))
    victim.eval()
    return victim

def get_poisoned_loaders():
    from torch.utils.data import DataLoader
    train_loader, train_dataset = get_dataloader(config, split='train', shuffle=True)
    _, test_dataset = get_dataloader(config, split='test', shuffle=False)

    train_dataset, _ = poison_dataset(config, train_dataset)
    test_dataset, _ = poison_dataset(config, test_dataset, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    return train_loader, train_dataset, test_loader, test_dataset

def pooled_features(model, x):
    N, C, T, V, M = x.shape
    h = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
    h = model.embed(h)
    h = model.block1(h, model.A)
    h = model.block2(h, model.A)
    h = model.block3(h, model.A)
    h = model.block4(h, model.A)
    h = model.block5(h, model.A)
    h = model.pool(h).squeeze(-1).squeeze(-1)
    
    # FIX: Isolate the primary actor (M=0) instead of averaging with the empty zero-matrix
    h = h.view(N, M, -1)[:, 0, :]
    return h

def collect_test_arrays(model, test_loader, device):
    all_x, all_labels, all_poison = [], [], []
    all_logits, all_features, all_pred = [], [], []

    model.eval()
    with torch.no_grad():
        for x, labels, poison_flags in test_loader:
            x = x.to(device)
            logits = model(x)
            feats = pooled_features(model, x)

            all_x.append(x.cpu())
            all_labels.append(labels)
            all_poison.append(poison_flags)
            all_logits.append(logits.cpu())
            all_features.append(feats.cpu())
            all_pred.append(logits.argmax(dim=1).cpu())

    return {
        'x': torch.cat(all_x, dim=0),
        'labels': torch.cat(all_labels, dim=0).numpy(),
        'poison_flags': torch.cat(all_poison, dim=0).numpy().astype(bool),
        'logits': torch.cat(all_logits, dim=0),
        'features': torch.cat(all_features, dim=0).numpy(),
        'pred': torch.cat(all_pred, dim=0).numpy(),
    }

def compute_fpr_at_tpr(scores, true_flags, tpr_target=0.95):
    fpr, tpr, _ = roc_curve(true_flags, scores)
    idx = np.searchsorted(tpr, tpr_target)
    idx = min(idx, len(fpr) - 1)
    return fpr[idx] * 100

def compute_precision_at_budget(scores, true_flags):
    n_poison = int(np.sum(true_flags))
    if n_poison == 0: return 0.0
    top_idx = np.argsort(scores)[-n_poison:]
    return np.sum(np.asarray(true_flags)[top_idx]) / n_poison * 100

def summarize(name, scores, true_flags, loc_iou="N/A"):
    scores = np.asarray(scores, dtype=float)
    true_flags = np.asarray(true_flags).astype(int)
    if len(np.unique(true_flags)) < 2:
        return {'method': name, 'auroc': float('nan'), 'auprc': float('nan'),
                'fpr@95tpr': float('nan'), 'precision@budget': float('nan'), 'loc_iou': "N/A"}
    return {
        'method': name,
        'auroc': roc_auc_score(true_flags, scores) * 100,
        'auprc': average_precision_score(true_flags, scores) * 100,
        'fpr@95tpr': compute_fpr_at_tpr(scores, true_flags, 0.95),
        'precision@budget': compute_precision_at_budget(scores, true_flags),
        'loc_iou': loc_iou
    }