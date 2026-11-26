
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
