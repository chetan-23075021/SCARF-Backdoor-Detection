"""
Data loading and preprocessing for NTU RGB+D
"""

import numpy as np
import pickle
import os
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import interpolate

class NTUDataset(Dataset):
    """NTU RGB+D Dataset"""
    def __init__(self, data_path, split='train', transform=None, subset_size=None):
        self.data_path = data_path
        self.split = split
        self.transform = transform
        
        # Load data
        data_file = os.path.join(data_path, f'{split}_data.npy')
        label_file = os.path.join(data_path, f'{split}_label.pkl')
        
        print(f"Loading {split} data from: {data_file}")
        
        # Load data WITHOUT mmap_mode to make it writable
        self.data = np.load(data_file)  # Removed mmap_mode='r'
        
        # Load labels
        with open(label_file, 'rb') as f:
            loaded_data = pickle.load(f)
            if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                self.labels = loaded_data[1]
            else:
                self.labels = loaded_data
        
        # Take subset if specified
        if subset_size and len(self.data) > subset_size:
            print(f"Taking subset of {subset_size} samples from {len(self.data)}")
            indices = np.random.choice(len(self.data), subset_size, replace=False)
            self.data = self.data[indices]
            self.labels = self.labels[indices]
        
        print(f"Data loaded. Shape: {self.data.shape}")
        print(f"Labels loaded. Shape: {self.labels.shape}")
        print(f"Number of samples: {len(self.data)}")
        
        # Make sure data is writable for poisoning
        self.data = self.data.copy()  # Force a writable copy
        
        # Poison flags (will be set later)
        self.poison_flags = np.zeros(len(self.data), dtype=bool)
        
    def get_poison_flags(self):
        return self.poison_flags

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Ensure data is converted properly
        data_slice = self.data[idx].copy()  # Make a copy to avoid warnings
        x = torch.FloatTensor(data_slice)
        y = torch.LongTensor([self.labels[idx]])[0]
        p = torch.BoolTensor([self.poison_flags[idx]])[0]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y, p
    
    def set_poison_flags(self, poison_indices):
        self.poison_flags[poison_indices] = True


class ToTensor:
    """Convert numpy array to tensor"""
    def __call__(self, x):
        return torch.FloatTensor(x)


def get_dataloader(config, split='train', shuffle=True):
    """Get data loader for specified split"""
    print(f"\nCreating {split} dataloader...")
    
    dataset = NTUDataset(
        data_path=config.DATA_PATH,
        split=split,
        transform=ToTensor(),
        subset_size=config.SUBSET_SIZE if hasattr(config, 'SUBSET_SIZE') else None
    )
    
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty! No {split} samples found at {config.DATA_PATH}")
    
    # Reduce number of workers to avoid warnings
    num_workers = min(2, config.NUM_WORKERS)
    
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False  # Set to False to avoid warnings
    )
    
    print(f"Dataloader created with {len(dataset)} samples, {len(loader)} batches")
    
    return loader, dataset

def auto_align(data):
    """3D alignment for skeleton data"""
    data = data[np.newaxis, ...]
    N, C, T, V, M = data.shape
    s = data.transpose(0, 4, 2, 3, 1)
    
    center = s[0, 0, 0, 0].copy()
    s = s - center
    
    shoulder_left = s[0, 0, 0, 4]
    shoulder_right = s[0, 0, 0, 8]
    vec = shoulder_right - shoulder_left
    dx, dy, dz = vec
    
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        angle_y = 0
    else:
        angle_y = -np.arctan2(dz, dx)
    
    cos, sin = np.cos(angle_y), np.sin(angle_y)
    Ry = np.array([[cos, 0, -sin],
                   [0, 1, 0],
                   [sin, 0, cos]])
    
    s_reshaped = s.reshape(-1, 3)
    s_rotated = np.dot(s_reshaped, Ry.T)
    s = s_rotated.reshape(s.shape)
    
    data_aligned = s.transpose(0, 4, 2, 3, 1)
    return data_aligned[0]


def preprocess_raw_data(raw_dir, output_dir, target_frames=64):
    """Preprocess raw NTU skeleton files"""
    import glob
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Look for skeleton files
    pattern = os.path.join(raw_dir, '*.skeleton')
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} skeleton files")
    
    if len(files) == 0:
        print(f"No .skeleton files found in {raw_dir}")
        return
    
    train_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
    train_data, train_labels = [], []
    test_data, test_labels = [], []
    
    for i, f in enumerate(files):
        if i % 100 == 0:
            print(f"Processing file {i}/{len(files)}...")
            
        filename = os.path.basename(f)
        try:
            # Parse NTU filename format: S001C001P001R001A001.skeleton
            action_class = int(filename[17:20]) - 1
            subject_id = int(filename[9:12])
            
            # Read skeleton file
            data = read_skeleton_file(f)
            if data is None or data.shape[1] < 5:  # Skip files with too few frames
                continue
            
            # Align
            data = auto_align(data)
            
            # Resize to target_frames
            C, T, V, M = data.shape
            resized = np.zeros((C, target_frames, V, M))
            for c in range(C):
                for v in range(V):
                    for m in range(M):
                        if T > 1:
                            x_old = np.arange(T)
                            x_new = np.linspace(0, T-1, target_frames)
                            f_interp = interpolate.interp1d(x_old, data[c, :, v, m], 
                                                           kind='linear', 
                                                           fill_value='extrapolate')
                            resized[c, :, v, m] = f_interp(x_new)
                        else:
                            resized[c, :, v, m] = data[c, 0, v, m]
            
            if subject_id in train_subjects:
                train_data.append(resized)
                train_labels.append(action_class)
            else:
                test_data.append(resized)
                test_labels.append(action_class)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    # Save
    if len(train_data) > 0:
        print(f"Saving {len(train_data)} train samples...")
        np.save(os.path.join(output_dir, 'train_data.npy'), np.array(train_data))
        with open(os.path.join(output_dir, 'train_label.pkl'), 'wb') as f:
            pickle.dump((['']*len(train_labels), np.array(train_labels)), f)
        
        print(f"Saving {len(test_data)} test samples...")
        np.save(os.path.join(output_dir, 'test_data.npy'), np.array(test_data))
        with open(os.path.join(output_dir, 'test_label.pkl'), 'wb') as f:
            pickle.dump((['']*len(test_labels), np.array(test_labels)), f)
        
        print(f"Preprocessed {len(train_data)} train, {len(test_data)} test samples")
    else:
        print("No valid data was processed!")


def read_skeleton_file(file_path):
    """Read a single NTU skeleton file"""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        if len(lines) < 1:
            return None
        
        frame_count = int(lines[0])
        data = np.zeros((3, frame_count, 25, 2))
        
        line_idx = 1
        for f_idx in range(frame_count):
            if line_idx >= len(lines):
                break
            body_count = int(lines[line_idx])
            line_idx += 1
            
            for b_idx in range(body_count):
                if line_idx >= len(lines):
                    break
                line_idx += 1
                if line_idx >= len(lines):
                    break
                joint_count = int(lines[line_idx])
                line_idx += 1
                
                for j_idx in range(joint_count):
                    if line_idx >= len(lines):
                        break
                    coords = list(map(float, lines[line_idx].strip().split()))
                    if b_idx < 2 and j_idx < 25:
                        data[:, f_idx, j_idx, b_idx] = coords[:3]
                    line_idx += 1
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None