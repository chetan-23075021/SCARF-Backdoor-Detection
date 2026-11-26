"""
Main execution script for SCARF
"""

import os
import argparse
import torch
import copy
from config import config
from data_loader import get_dataloader, preprocess_raw_data
from train_victim import train_victim
from train_completion import train_completion
from inject_poison import poison_dataset
from detect import run_detection
from visualize import generate_all_figures
from utils import set_seed
from torch.utils.data import DataLoader

def main(args):
    """Main execution function"""
    set_seed(42)
    
    print("="*60)
    print("SCARF: Complete Training and Evaluation Pipeline")
    print("="*60)
    
    # Step 1: Preprocess raw data if needed
    if args.preprocess:
        print("\n[Step 1] Preprocessing raw data...")
        preprocess_raw_data(args.raw_dir, config.DATA_PATH)
    
    # Step 2: Load data
    print("\n[Step 2] Loading data...")
    train_loader, train_dataset = get_dataloader(config, split='train', shuffle=True)
    # val_loader, _ = get_dataloader(config, split='test', shuffle=False)
    # test_loader, test_dataset = get_dataloader(config, split='test', shuffle=False) 
    # import copy
    val_loader, val_dataset = get_dataloader(config, split='test', shuffle=False)
    test_dataset = copy.deepcopy(val_dataset)          # in-RAM copy, no second Drive read
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS)
    print(f"  Train samples: {len(train_dataset)}")
    
    # ==========================================
    # FIX: Step 3 and 4 have been swapped!
    # ==========================================
    
    # Step 3: Poison dataset FIRST
    if args.poison:
        print("\n[Step 3] Poisoning training dataset (For Victim & Completion Models)...")
        train_dataset, poison_indices_train = poison_dataset(config, train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
        print(f"  Poisoned {len(poison_indices_train)} training samples")
        
        print("\n[Step 3.5] Poisoning test dataset (For SCARF Detection)...")
        test_dataset, poison_indices_test = poison_dataset(config, test_dataset, is_test=True)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
        print(f"  Poisoned {len(poison_indices_test)} test samples")

    # Step 4: Train victim model SECOND (Now it learns the backdoor!)
    if args.train_victim:
        print("\n[Step 4] Training victim ST-GCN model...")
        victim_model = train_victim(train_loader, val_loader)

    # Step 5: Train completion model(s)
    if args.exp_contamination:
        print("\n[Step 5 - Exp 8] Running Completion Contamination Experiment...")
        
        # --- RAM SAVER: Destroy Step 2 datasets since Exp 8 doesn't use them ---
        import gc
        if 'train_dataset' in locals(): 
            del train_dataset, train_loader
        if 'test_dataset' in locals(): 
            del test_dataset, test_loader
        if 'val_dataset' in locals():
            del val_dataset, val_loader
        gc.collect()
        print(" Freed memory from initial dataloaders.")
        # ------------------------------------------------------------------------
        
        contamination_rates = [0.0, 0.01, 0.05, 0.10, 0.20]
        
        for rate in contamination_rates:
            print(f"\n" + "="*50)
            print(f"--- Training Completion Model with {rate*100}% Poisoning ---")
            print("="*50)
            
            orig_rate = config.POISON_RATE
            config.POISON_RATE = rate
            
            # Load ONE fresh copy. (The Linux OS caches the Drive file, making this fast).
            contam_loader, contam_dataset = get_dataloader(config, split='train', shuffle=True)
            
            if rate > 0:
                contam_dataset, _ = poison_dataset(config, contam_dataset)
                # FIX: Removed the local 'from torch.utils.data import DataLoader' here
                contam_loader = DataLoader(contam_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
                
            train_completion(contam_loader, contamination_rate=rate)
            
            config.POISON_RATE = orig_rate
            
            # --- RAM SAVER: Destroy this dataset before the next loop starts ---
            del contam_dataset, contam_loader
            gc.collect()
            
    elif args.train_completion:
        print("\n[Step 5] Training standard clean completion model...")
        completion_model = train_completion(train_loader, contamination_rate=0.0)
    
    # Step 6: Run SCARF detection
    if args.detect:
        print("\n[Step 6] Running SCARF detection...")
        if not hasattr(test_loader.dataset, 'poison_flags'):
             print("[Warning] Running detection on unpoisoned test set. Metrics will be 0.")
             
        scores, poison_flags, regions, detector = run_detection(test_loader, train_loader)
        
        from evaluation import compute_detection_metrics, compute_localization_metrics
        
        if hasattr(test_loader.dataset, 'poison_flags'):
            true_flags = test_loader.dataset.poison_flags[:len(scores)]
        else:
            true_flags = [0] * len(scores)
            
        metrics = compute_detection_metrics(scores, true_flags)
        print("\n📊 SCARF Detection Performance:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.2f}%")
        
        true_regions = getattr(test_loader.dataset, 'true_regions', []) 
        if not true_regions:
            print("[Warning] No ground-truth 'true_regions' found in test dataset. Localization will be 0.")
            
        loc_metrics = compute_localization_metrics(regions, true_regions)
        print(f"\n📍 Localization Performance:")
        for key, value in loc_metrics.items():
            print(f"  {key}: {value:.2f}%")
        
        results = {
            'scores': scores,
            'poison_flags': poison_flags,
            'true_flags': true_flags,
            'metrics': metrics,
            'loc_metrics': loc_metrics,
            'config': vars(args)
        }
        
        import pickle
        with open(os.path.join(config.CHECKPOINT_DIR, 'detection_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        print(f"\n💾 Results saved to {config.CHECKPOINT_DIR}/detection_results.pkl")

    # Step 7: Visualize
    if args.visualize:
        print("\n[Step 7] Generating paper figures...")
        from visualize import generate_all_figures
        
        sc = scores if 'scores' in locals() else None
        pf = poison_flags if 'poison_flags' in locals() else None
        sig = detector.signatures if ('detector' in locals() and hasattr(detector, 'signatures')) else None

        generate_all_figures(
            ilp_scores=sc,
            poison_flags=pf,
            signatures=sig
        )
        print(f"✅ Figures saved to {config.FIGURE_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCARF: Complete Pipeline")
    parser.add_argument('--preprocess', action='store_true', help='Preprocess raw data')
    parser.add_argument('--raw_dir', type=str, default='./raw_ntu', help='Raw data directory')
    parser.add_argument('--train_victim', action='store_true', help='Train victim model')
    parser.add_argument('--train_completion', action='store_true', help='Train standard completion model')
    parser.add_argument('--exp_contamination', action='store_true', help='Run Completion Contamination Experiment (0-20%)')
    parser.add_argument('--poison', action='store_true', help='Poison dataset')
    parser.add_argument('--detect', action='store_true', help='Run detection')
    parser.add_argument('--visualize', action='store_true', help='Generate figures')
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
    else:
        main(args)






        
