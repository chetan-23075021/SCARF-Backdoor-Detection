"""
Train completion model for SCARF with Auto-Resume Checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import CompletionModel
from config import config
import os
import random
from utils import save_checkpoint, load_checkpoint  # <--- IMPORT ADDED

def train_completion(train_loader, contamination_rate=0.0):
    """
    Train the self-supervised completion model.
    contamination_rate: Used to tag model saves for Experiment 8.
    """
    device = torch.device(config.DEVICE)
    model = CompletionModel(
        in_channels=config.INPUT_CHANNELS
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE_COMPLETION)
    criterion = nn.MSELoss()
    
    # Prefix for saving models during the Completion Contamination experiment
    prefix = f"completion_contam_{int(contamination_rate*100)}pct" if contamination_rate > 0 else "completion"
    
    start_epoch = 0
    latest_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'{prefix}_latest_checkpoint.pth')
    
    # =========================================================
    # RESUME LOGIC: Check if a previous run was interrupted
    # =========================================================
    if os.path.exists(latest_checkpoint_path):
        print(f"\n🔄 Found interrupted completion training! Resuming from: {latest_checkpoint_path}")
        last_epoch = load_checkpoint(model, optimizer, latest_checkpoint_path, device=config.DEVICE)
        start_epoch = last_epoch + 1
        print(f"▶️ Fast-forwarded to Epoch {start_epoch}. Resuming training...\n")
    # =========================================================
    
    for epoch in range(start_epoch, config.EPOCHS_COMPLETION):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, _, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Remove person dimension for completion
            if len(data.shape) == 5:  # (N, C, T, V, M)
                data = data[..., 0]  # Average over persons: (N, C, T, V)
                
            # --- Self-Supervised Masking Logic ---
            masked_data = data.clone()
            N, C, T, V = masked_data.shape
            
            # Generate random mask bounds for the batch
            t_window = random.randint(5, min(15, T))
            v_window = random.randint(3, min(7, V))
            t_start = random.randint(0, max(1, T - t_window))
            v_start = random.randint(0, max(1, V - v_window))
            
            # Apply hard mask (zero out the region) to force the model to learn context
            masked_data[:, :, t_start:t_start+t_window, v_start:v_start+v_window] = 0.0
            
            # Forward pass using the MASKED data
            recon, _ = model(masked_data)
            
            # Compute loss against the ORIGINAL UNMASKED data to measure reconstruction quality
            loss = criterion(recon, data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch} Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} Avg Reconstruction Loss (MSE): {avg_loss:.4f}')
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 
                      os.path.join(config.CHECKPOINT_DIR, f'{prefix}_epoch{epoch}.pth'))
                      
        # =========================================================
        # SAVE STATE: Save progress at the end of every single epoch
        # =========================================================
        save_checkpoint(model, optimizer, epoch, latest_checkpoint_path)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, f'{prefix}_final.pth'))
    return model