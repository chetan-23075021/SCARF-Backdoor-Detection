import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from models import STGCN

def _lga_loss(ce_per_sample, gamma):
    return torch.abs(ce_per_sample - gamma) + gamma

def run_abl(train_loader, device=None, isolation_epochs=10, total_epochs=20, gamma=0.5, isolation_ratio=None, lr=0.01):
    # FIX: Dynamically link isolation ratio to the actual config poison rate
    if isolation_ratio is None:
        isolation_ratio = config.POISON_RATE 
        
    device = device or torch.device(config.DEVICE)
    model = STGCN(num_classes=config.NUM_CLASSES, num_joints=config.NUM_JOINTS, in_channels=config.INPUT_CHANNELS).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0004)
    ce = nn.CrossEntropyLoss(reduction='none')

    n_total = len(train_loader.dataset)
    final_isolation_loss = np.zeros(n_total, dtype=float)
    all_poison_flags = np.zeros(n_total, dtype=bool)

    bookkeeping_loader = DataLoader(train_loader.dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"[ABL] Stage 1: isolation training for {isolation_epochs} epochs (LGA loss, gamma={gamma})")
    for epoch in range(isolation_epochs):
        model.train()
        running = 0.0
        for data, labels, _ in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(data)
            per_sample = ce(logits, labels)
            loss = _lga_loss(per_sample, gamma).mean()
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"  [ABL isolation] epoch {epoch} loss={running/len(train_loader):.4f}")

    model.eval()
    ptr = 0
    with torch.no_grad():
        for data, labels, poison_flags in bookkeeping_loader:
            data, labels = data.to(device), labels.to(device)
            logits = model(data)
            per_sample = ce(logits, labels).cpu().numpy()
            b = len(per_sample)
            final_isolation_loss[ptr:ptr+b] = per_sample
            all_poison_flags[ptr:ptr+b] = poison_flags.numpy().astype(bool)
            ptr += b

    n_isolate = max(1, int(n_total * isolation_ratio))
    isolated_idx = np.argsort(final_isolation_loss)[:n_isolate]
    isolated_mask = np.zeros(n_total, dtype=bool)
    isolated_mask[isolated_idx] = True

    print(f"[ABL] Stage 2: unlearning for {total_epochs - isolation_epochs} more epochs on {n_isolate} isolated samples")
    for epoch in range(isolation_epochs, total_epochs):
        model.train()
        ptr = 0
        for data, labels, _ in bookkeeping_loader:
            b = len(data)
            batch_isolated = torch.from_numpy(isolated_mask[ptr:ptr+b]).to(device)
            ptr += b
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(data)
            per_sample = ce(logits, labels)
            sign = torch.where(batch_isolated, -1.0, 1.0)
            loss = (per_sample * sign).mean()
            loss.backward()
            optimizer.step()
        print(f"  [ABL unlearning] epoch {epoch} done")

    scores = -final_isolation_loss
    return scores, all_poison_flags, model