"""
Diagnostic: compare victim model confidence on poisoned vs clean test samples.
Single forward pass only -- no training, no backward pass.
"""
import torch
import numpy as np
from config import config
from data_loader import get_dataloader
from inject_poison import poison_dataset
from models import STGCN
from torch.utils.data import DataLoader

device = torch.device(config.DEVICE)

_, test_dataset = get_dataloader(config, split='test', shuffle=False)
test_dataset, _ = poison_dataset(config, test_dataset, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                          shuffle=False, num_workers=config.NUM_WORKERS)

victim = STGCN(num_classes=config.NUM_CLASSES).to(device)
victim.load_state_dict(torch.load(f'{config.CHECKPOINT_DIR}/best_victim.pth', map_location=device))
victim.eval()

poison_conf, clean_conf, poison_hit_target = [], [], []

with torch.no_grad():
    for x, labels, poison_flags in test_loader:
        x = x.to(device)
        probs = torch.softmax(victim(x), dim=1)
        conf, pred = probs.max(dim=1)
        flags = poison_flags.numpy() if torch.is_tensor(poison_flags) else np.array(poison_flags)
        conf, pred = conf.cpu().numpy(), pred.cpu().numpy()

        for c, p, f in zip(conf, pred, flags):
            if f:
                poison_conf.append(c)
                poison_hit_target.append(int(p == config.TARGET_CLASS))
            else:
                clean_conf.append(c)

poison_conf, clean_conf = np.array(poison_conf), np.array(clean_conf)
print(f"Poisoned samples: {len(poison_conf)}")
print(f"  Mean/median max-softmax confidence: {poison_conf.mean():.4f} / {np.median(poison_conf):.4f}")
print(f"  % predicted as TARGET_CLASS: {np.mean(poison_hit_target)*100:.2f}%")
print(f"\nClean samples: {len(clean_conf)}")
print(f"  Mean/median max-softmax confidence: {clean_conf.mean():.4f} / {np.median(clean_conf):.4f}")