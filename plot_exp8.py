"""
Automated Subprocess Evaluator for Experiment 8
"""
import os
import sys
import subprocess
import shutil
import pickle
import csv
import matplotlib.pyplot as plt

print("="*60)
print("🛠️  Applying Auto-Fixes to Configuration...")
# 1. Physically fix config.py on the hard drive to stop the 5-hour freezing
if os.path.exists('config.py'):
    with open('config.py', 'r') as f:
        lines = f.readlines()
    with open('config.py', 'w') as f:
        for line in lines:
            if 'NUM_WORKERS' in line and '=' in line:
                f.write("    NUM_WORKERS = 0  # AUTO-FIXED TO PREVENT COLAB DEADLOCKS\n")
            else:
                f.write(line)
    print("✅ Fixed config.py: NUM_WORKERS is now 0!")

def run_eval():
    # Force the GPU to be visible to the background processes
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    ckpt_dir = './checkpoints/NTU60'
    target_path = os.path.join(ckpt_dir, 'completion_final.pth')
    backup_path = os.path.join(ckpt_dir, 'completion_final_backup.pth')
    
    rates = [0, 1, 5, 10, 20]
    aurocs, auprcs, fprints = [], [], []

    # Backup the 0% clean model
    if os.path.exists(target_path) and not os.path.exists(backup_path):
        shutil.copy(target_path, backup_path)

    for rate in rates:
        print(f"\n{'='*50}\n🚀 Evaluating {rate}% Contamination Model\n{'='*50}")
        
        # Load the right checkpoint
        source = backup_path if rate == 0 else os.path.join(ckpt_dir, f'completion_contam_{rate}pct_final.pth')
        if not os.path.exists(source):
            print(f"❌ Missing file: {source}")
            continue
            
        shutil.copy(source, target_path)
        
        # Run main.py safely
        cmd = [sys.executable, 'main.py', '--poison', '--detect']
        print(f"⏳ Executing detection... (This will take ~90 seconds)")
        
        subprocess.run(cmd, env=env, check=True)
        
        # Grab the results
        res_file = os.path.join(ckpt_dir, 'detection_results.pkl')
        with open(res_file, 'rb') as f:
            res = pickle.load(f)
            
        metrics = res.get('metrics', {})
        auc = metrics.get('auroc', metrics.get('AUROC', 0.0))
        prc = metrics.get('auprc', metrics.get('AUPRC', 0.0))
        fpr = metrics.get('fpr@95', metrics.get('fpr@95tpr', metrics.get('FPR@95TPR', 0.0)))
        
        aurocs.append(auc)
        auprcs.append(prc)
        fprints.append(fpr)
        
        print(f"\n✅ Result for {rate}% -> AUROC: {auc:.2f}% | AUPRC: {prc:.2f}%")

    # Restore the original model
    if os.path.exists(backup_path):
        shutil.copy(backup_path, target_path)
    
    # Save CSV
    os.makedirs('./results', exist_ok=True)
    with open('./results/experiment8_contamination_results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Contamination_Rate(%)', 'AUROC(%)', 'AUPRC(%)', 'FPR@95TPR(%)'])
        for r, a, p, fpr in zip(rates, aurocs, auprcs, fprints):
            w.writerow([r, f"{a:.2f}", f"{p:.2f}", f"{fpr:.2f}"])
            
    # Draw Graph
    plt.figure(figsize=(8, 5.5))
    plt.plot(rates, aurocs, marker='o', linewidth=2.5, markersize=8, color='#1f77b4', label='AUROC (%)')
    plt.plot(rates, auprcs, marker='s', linewidth=2.0, markersize=7, color='#2ca02c', linestyle='--', label='AUPRC (%)')
    plt.fill_between(rates, aurocs, alpha=0.1, color='#1f77b4')
    plt.title('SCARF Robustness to Training Contamination', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Poison Contamination in Completion Training Data (%)', fontsize=11)
    plt.ylabel('Detection Metric (%)', fontsize=11)
    plt.ylim(0, 105)
    plt.xticks(rates)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower left')
    
    for i, txt in enumerate(aurocs):
        plt.annotate(f"{txt:.1f}%", (rates[i], aurocs[i]), xytext=(0, 8), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
        
    os.makedirs('./figures/NTU60/exp8', exist_ok=True)
    save_path = './figures/NTU60/exp8/exp8_contamination_robustness.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\n🎉 Experiment 8 Complete! Graph saved to {save_path}")

if __name__ == '__main__':
    run_eval()