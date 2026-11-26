"""
Configuration settings for SCARF
"""

import os
import torch

class Config:
    # Paths
    DATA_PATH = '/content/drive/MyDrive/BTP/Skeleton project/data/processed/NTU60'
    CHECKPOINT_DIR = './checkpoints/NTU60'
    FIGURE_DIR = './figures/NTU60'
    SUBSET_SIZE = None
    
    # Dataset settings (FIXED: Properly aligned with NTU-60)
    DATASET = 'NTU60'  # Fixed from UAV_HUMAN
    NUM_CLASSES = 60
    NUM_JOINTS = 25 
    NUM_FRAMES = 64
    INPUT_CHANNELS = 3
    
    # Training settings
    BATCH_SIZE = 32
    NUM_WORKERS = 0
    EPOCHS_VICTIM = 50
    EPOCHS_COMPLETION = 30
    LEARNING_RATE_VICTIM = 0.01
    LEARNING_RATE_COMPLETION = 0.001
    
    # Attack settings
    POISON_RATE = 0.10  # Increased to guarantee the model learns the backdoor
    TARGET_CLASS = 0
    TRIGGER_TYPE = 'PSBA'  
    # Low Poisoning Rate Experiment Settings
    LOW_POISON_RATES = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]
    # PSBA Attack settings
    PSBA_TRIGGER = 'nodding' 
    TRIGGER_INTENSITY = 0.8  # Increased significantly so it survives ST-GCN Dropout
    TRIGGER_DURATION = 16  
    
    # SCARF detection settings
    NUM_CANDIDATES = 10  
    WINDOW_SIZES = [5, 10, 15]  
    CONSENSUS_SEEDS = 3  
    CONTEXT_PERTURBATIONS = 3  
    
    # Device
    DEVICE = 'cuda' 
    
    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

config = Config()