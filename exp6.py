import os
import copy
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. Project directory setup
project_dir = "/content/drive/MyDrive/BTP/Skeleton project/SCARF_Complete"
if os.path.exists(project_dir):
    os.chdir(project_dir)

from config import config
config.NUM_WORKERS = 0  
config.DEVICE = 'cuda'

from data_loader import get_dataloader
from detect import run_detection
from torch.utils.data import DataLoader

print("="*60)
print("🚀 EXP 6: FALSE-POSITIVE & NOISE ROBUSTNESS EVALUATION (1500 SAMPLES)")
print("="*60)

# Load clean subset locked at 1500 samples for methodological consistency
train_loader, train_dataset = get_dataloader(config, split='train', shuffle=True)
_, val_dataset = get_dataloader(config, split='test', shuffle=False)

base_clean_dataset = copy.deepcopy(val_dataset)
base_clean_dataset.data = base_clean_dataset.data[:1500]
base_clean_dataset.labels = base_clean_dataset.labels[:1500]
# Ensure no poisoned flags exist
base_clean_dataset.poison_flags = [0] * 1500
base_clean_dataset.true_regions = [None] * 1500

# Noise sweep parameters (Standard deviation of Gaussian kinematic jitter)
noise_levels = [0.0, 0.02, 0.05, 0.10, 0.15]
mean_scores = []
max_scores = []

os.makedirs('./results/exp6_data', exist_ok=True)

for noise in noise_levels:
    print(f"\n{'='*50}\n🚀 Evaluating Kinematic Noise Level: {noise}\n{'='*50}")
    
    # Inject pure Gaussian noise into the clean data to simulate sensor jitter
    noisy_dataset = copy.deepcopy(base_clean_dataset)
    if noise > 0:
        jitter = np.random.randn(*noisy_dataset.data.shape) * noise
        noisy_dataset.data = noisy_dataset.data + jitter
        
    noisy_loader = DataLoader(noisy_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Run SCARF Detection
    scores, _, _, _ = run_detection(noisy_loader, train_loader)
    
    # We only care about the scores generated on this clean data
    avg_score = np.mean(scores)
    peak_score = np.max(scores)
    
    mean_scores.append(avg_score)
    max_scores.append(peak_score)
    
    print(f"✅ RESULT FOR NOISE {noise} -> Mean SCARF Score: {avg_score:.4f} | Max Score: {peak_score:.4f}")

# Generate CSV and Final Graph
csv_path = './results/experiment6_false_positives.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Noise_Level(StdDev)', 'Mean_Anomaly_Score', 'Max_Anomaly_Score'])
    for n, avg, mx in zip(noise_levels, mean_scores, max_scores):
        writer.writerow([n, f"{avg:.4f}", f"{mx:.4f}"])

# The critical threshold (from earlier clean baseline runs, poisoned samples sit >12.0)
safe_threshold = 8.0 

plt.figure(figsize=(8, 5.5))
plt.plot(noise_levels, mean_scores, marker='o', linewidth=2.5, color='#2ca02c', label='Mean Anomaly Score')
plt.plot(noise_levels, max_scores, marker='s', linewidth=2.0, color='#ff7f0e', linestyle='--', label='Max Anomaly Score (Worst Case)')
plt.axhline(y=safe_threshold, color='red', linestyle=':', linewidth=2, label='Typical Poison Detection Threshold')

plt.fill_between(noise_levels, mean_scores, alpha=0.1, color='#2ca02c')

plt.title('Exp 6: SCARF False-Positive Robustness to Natural Jitter (1.5k Subset)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Kinematic Noise Level (Standard Deviation)', fontsize=11)
plt.ylabel('SCARF Logit-Margin Score', fontsize=11)
plt.ylim(0, max(15, max(max_scores) + 2))
plt.xticks(noise_levels)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left')

for i, txt in enumerate(mean_scores):
    plt.annotate(f"{txt:.2f}", (noise_levels[i], mean_scores[i]), xytext=(0, 8), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
    
os.makedirs('./figures/NTU60/exp6', exist_ok=True)
save_path = './figures/NTU60/exp6/exp6_false_positive_noise.png'
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"\n🎉 EXP 6 COMPLETE! Graph saved to {save_path}")