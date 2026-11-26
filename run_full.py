
import argparse
import main as main_module

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
