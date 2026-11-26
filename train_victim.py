"""
Train victim ST-GCN model with Auto-Resume Checkpointing and ASR Tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import STGCN
from config import config
import os
from visualize import plot_metrics
from utils import save_checkpoint, load_checkpoint  

def train_victim(train_loader, val_loader):
    """Train the victim ST-GCN model"""
    device = torch.device(config.DEVICE)
    
    model = STGCN(
        num_classes=config.NUM_CLASSES, 
        num_joints=config.NUM_JOINTS, 
        in_channels=config.INPUT_CHANNELS
    ).to(config.DEVICE)
    
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE_VICTIM,
                          momentum=0.9, weight_decay=0.0004)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    start_epoch = 0
    latest_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'latest_victim_checkpoint.pth')
    
    # =========================================================
    # RESUME LOGIC: Check if a previous run was interrupted
    # =========================================================
    if os.path.exists(latest_checkpoint_path):
        print(f"\n🔄 Found interrupted training! Resuming from: {latest_checkpoint_path}")
        last_epoch = load_checkpoint(model, optimizer, latest_checkpoint_path, device=config.DEVICE)
        start_epoch = last_epoch + 1  
        
        for _ in range(start_epoch):
            scheduler.step()
            
        print(f"▶️ Fast-forwarded to Epoch {start_epoch}. Resuming training...\n")
    # =========================================================
    
    train_loss_history = []
    train_acc_history = []
    epochs_list = []
    
    for epoch in range(start_epoch, config.EPOCHS_VICTIM):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # ASR Tracking Variables
        poison_correct = 0
        poison_total = 0
        
        for batch_idx, (data, labels, poison_flags) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            poison_flags = poison_flags.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            
            # Overall Accuracy Math
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Isolated ASR Math
            p_mask = (poison_flags == 1)
            if p_mask.sum() > 0:
                poison_total += p_mask.sum().item()
                poison_correct += predicted[p_mask].eq(labels[p_mask]).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch} Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}')
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        train_asr = (100. * poison_correct / poison_total) if poison_total > 0 else 0.0
        
        print(f'Epoch {epoch} Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | Train ASR: {train_asr:.2f}%')
        
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(train_acc)
        epochs_list.append(epoch)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, labels, _ in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f'Epoch {epoch} Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, 'best_victim.pth'))
            print(f'  Saved best model with acc {val_acc:.2f}%')
            
        # =========================================================
        # SAVE STATE: Save progress at the end of every single epoch
        # =========================================================
        save_checkpoint(model, optimizer, epoch, latest_checkpoint_path)
        
        scheduler.step()
    
    print(f"Best validation accuracy: {best_acc:.2f}%")
    
    try:
        plot_metrics(
            dataset_name=config.DATASET, 
            current_epoch=config.EPOCHS_VICTIM, 
            phase="train", 
            train_mode="victim_training", 
            epochs_list=epochs_list, 
            acc_list=train_acc_history, 
            loss_list=train_loss_history
        )
    except Exception as e:
        print(f"[Visualization Warning] Failed to plot training curve: {e}")
        
    return model