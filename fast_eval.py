
import os
import sys
import subprocess
import shutil

# Auto-Fix Config exactly as before
with open('config.py', 'r') as f_conf:
    lines = f_conf.readlines()
with open('config.py', 'w') as f_conf:
    for line in lines:
        if 'NUM_WORKERS' in line and '=' in line:
            f_conf.write("    NUM_WORKERS = 0\n")
        else:
            f_conf.write(line)

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"
rates = [0, 1, 5, 10, 20]

wrapper = """
import sys
import argparse
import copy
import main as main_module
from data_loader import get_dataloader

original_get_dataloader = get_dataloader
def fast_dataloader(config, split, shuffle=False):
    loader, dataset = original_get_dataloader(config, split, shuffle)
    if split == 'test':
        # TRUNCATE to 1000 samples for lightning-fast testing
        dataset.data = dataset.data[:1000]
        dataset.labels = dataset.labels[:1000]
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    return loader, dataset

main_module.get_dataloader = fast_dataloader

parser = argparse.ArgumentParser()
parser.add_argument('--preprocess', action='store_true')
parser.add_argument('--raw_dir', type=str, default='./raw_ntu')
parser.add_argument('--train_victim', action='store_true')
parser.add_argument('--train_completion', action='store_true')
parser.add_argument('--exp_contamination', action='store_true')
parser.add_argument('--poison', action='store_true')
parser.add_argument('--detect', action='store_true')
parser.add_argument('--visualize', action='store_true')

args = parser.parse_args(['--poison', '--detect'])
main_module.main(args)
"""

with open('run_subset.py', 'w') as f_wrap:
    f_wrap.write(wrapper)

ckpt_dir = './checkpoints/NTU60'
os.makedirs('./results/exp8_data', exist_ok=True)

for rate in rates:
    print(f"\n{'='*50}\n🚀 Evaluating {rate}% Contamination Model\n{'='*50}")
    
    source = os.path.join(ckpt_dir, 'completion_final_backup.pth') if rate == 0 else os.path.join(ckpt_dir, f'completion_contam_{rate}pct_final.pth')
    if not os.path.exists(source):
        print(f"❌ Missing file: {source}")
        continue
        
    shutil.copy(source, os.path.join(ckpt_dir, 'completion_final.pth'))
    
    subprocess.run([sys.executable, 'run_subset.py'], env=env, check=True)
    shutil.copy(os.path.join(ckpt_dir, 'detection_results.pkl'), f'./results/exp8_data/res_{rate}.pkl')

print("\n✅ COMPLETED FAST EVALUATION!")
