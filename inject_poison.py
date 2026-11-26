"""
Poison injection functions for different trigger types, including Adaptive Attacks
"""

import numpy as np

def inject_distributed_trigger(data, num_regions=3):
    """Inject distributed trigger across multiple regions"""
    poisoned = data.copy()
    N, C, T, V, M = poisoned.shape
    regions = []
    
    for i in range(N):
        min_t, max_t = T, 0
        min_v, max_v = V, 0
        for _ in range(num_regions):
            t_start = np.random.randint(0, max(1, T-5))
            v_start = np.random.randint(0, max(1, V-3))
            # FIX: Only inject to primary actor (m=0) and increase magnitude
            poisoned[i, :, t_start:t_start+3, v_start:v_start+2, 0] += 0.3 
            
            min_t, max_t = min(min_t, t_start), max(max_t, t_start+3)
            min_v, max_v = min(min_v, v_start), max(max_v, v_start+2)
            
        regions.append((min_t, max_t, min_v, max_v))
        
    return poisoned, regions


def inject_sample_specific_trigger(data, dataset='NTU60'):
    """Adaptive Attack: Trigger varies per sample"""
    poisoned = data.copy()
    N, C, T, V, M = poisoned.shape
    regions = []
    
    t_center = T // 2
    t_start, t_end = max(0, t_center - 5), min(T, t_center + 5)
    
    joints = [7, 8, 9, 10] if dataset == 'UV_HUMAN' else [10, 11, 12, 13]
    v_start, v_end = min(joints), max(joints) + 1
    
    for i in range(N):
        sample_variance = np.var(data[i, :, :, joints, 0])
        # FIX: Multiply intensity to survive pooling
        intensity_multiplier = 0.5 + (sample_variance * 1.0) 
        
        for t in range(t_start, t_end):
            pattern = np.array([
                np.sin(t) * intensity_multiplier,
                np.cos(t) * intensity_multiplier,
                (np.sin(t) + np.cos(t)) * intensity_multiplier
            ])
            
            for j in joints:
                if j < V:
                    # FIX: Apply only to m=0
                    poisoned[i, :, t, j, 0] += pattern
                    
        regions.append((t_start, t_end, v_start, v_end))
        
    return poisoned, regions


def inject_anti_transfer_trigger(data, dataset='NTU60'):
    """Adaptive Attack: Absolute coordinate trigger"""
    poisoned = data.copy()
    N, C, T, V, M = poisoned.shape
    regions = []
    
    t_start, t_end = 0, T // 3
    joints = [0, 1, 2] if dataset == 'UV_HUMAN' else [0, 1, 2, 3]
    v_start, v_end = min(joints), max(joints) + 1
    
    for i in range(N):
        global_anchor_offset = 0.5 - np.mean(data[i, :, t_start:t_end, joints, 0]) 
        
        for t in range(t_start, t_end):
            for j in joints:
                if j < V:
                    poisoned[i, :, t, j, 0] += (global_anchor_offset * 0.8) # Increased
                    
        regions.append((t_start, t_end, v_start, v_end))
        
    return poisoned, regions


def inject_psba_trigger_real(data, trigger_type='nodding', intensity=0.1, dataset='NTU60'):
    """Real PSBA-style physically plausible trigger implementation"""
    
    # FIX: Multiplied baseline sine/cosine scales by 10 to simulate 40-80cm actual space movement
    patterns = {
        'nodding': lambda t: np.array([0.5 * np.sin(2*np.pi*t/15), 0.3 * np.cos(2*np.pi*t/15), 0.8 * np.sin(2*np.pi*t/10)]) * intensity,
        'bending': lambda t: np.array([0.2 * np.sin(2*np.pi*t/20), 1.0 * np.sin(2*np.pi*t/25), 0.3 * np.cos(2*np.pi*t/20)]) * intensity,
        'crossing_hands': lambda t: np.array([1.2 * np.sin(2*np.pi*t/12), 0.5 * np.cos(2*np.pi*t/12), 0.4 * np.sin(2*np.pi*t/8)]) * intensity,
        'kicking': lambda t: np.array([0.8 * np.sin(2*np.pi*t/10), 1.5 * np.sin(2*np.pi*t/12), 0.6 * np.cos(2*np.pi*t/10)]) * intensity
    }
    
    if dataset == 'UV_HUMAN':
        joints_map = {
            'nodding': [0, 1, 2, 3, 4], 
            'bending': [5, 6, 11, 12], 
            'crossing_hands': [7, 8, 9, 10], 
            'kicking': [13, 14, 15, 16] 
        }
    else:
        joints_map = {
            'nodding': [0, 1, 2, 3], 
            'bending': [4, 5, 6, 7, 8, 9], 
            'crossing_hands': [10, 11, 12, 13, 14, 15], 
            'kicking': [16, 17, 18, 19, 20, 21] 
        }
    
    joints = joints_map.get(trigger_type, joints_map['nodding'])
    pattern_fn = patterns.get(trigger_type, patterns['nodding'])
    
    poisoned = data.copy()
    N, C, T, V, M = poisoned.shape
    regions = []
    
    t_center = T // 2
    t_start = max(0, t_center - 8)
    t_end = min(T, t_center + 8)
    v_start, v_end = min(joints), max(joints) + 1
    
    for i in range(N):
        for t in range(t_start, t_end):
            # FIX: Flatten the pattern so it applies strictly to a single person's joints
            pattern = pattern_fn(t - t_start).flatten()
            for j in joints:
                if j < V:
                    # FIX: Isolate injection to M=0 (the primary skeleton)
                    poisoned[i, :, t, j, 0] += pattern
        regions.append((t_start, t_end, v_start, v_end))
    
    return poisoned, regions

def inject_optimization_trigger(data, epsilon=0.5):
    """Inject optimization-based (noise) trigger"""
    # FIX: Increased epsilon significantly
    noise = np.random.randn(*data.shape) * epsilon
    noise = np.clip(noise, -epsilon, epsilon)
    # Zero out noise for M=1
    noise[:, :, :, :, 1] = 0.0
    N, C, T, V, M = data.shape
    regions = [(0, T, 0, V) for _ in range(N)]
    return data + noise, regions


def poison_dataset(config, train_dataset, is_test=False):
    """Poison dataset and map ground-truth regions"""
    n_samples = len(train_dataset.data)
    n_poison = max(1, int(n_samples * config.POISON_RATE)) if config.POISON_RATE > 0 else 0
    
    if n_poison == 0:
        print(f"Skipping poisoning (Rate is {config.POISON_RATE})")
        train_dataset.true_regions = [None] * n_samples
        return train_dataset, []
    
    np.random.seed(42)
    poison_indices = np.random.choice(n_samples, n_poison, replace=False)
    data_to_poison = train_dataset.data[poison_indices].copy()
    
    if config.TRIGGER_TYPE == 'PSBA':
        poisoned_subset, injected_regions = inject_psba_trigger_real(data_to_poison, config.PSBA_TRIGGER, intensity=config.TRIGGER_INTENSITY, dataset=config.DATASET)
    elif config.TRIGGER_TYPE == 'distributed':
        poisoned_subset, injected_regions = inject_distributed_trigger(data_to_poison)
    elif config.TRIGGER_TYPE == 'sample_specific':
        poisoned_subset, injected_regions = inject_sample_specific_trigger(data_to_poison, dataset=config.DATASET)
    elif config.TRIGGER_TYPE == 'anti_transfer':
        poisoned_subset, injected_regions = inject_anti_transfer_trigger(data_to_poison, dataset=config.DATASET)
    elif config.TRIGGER_TYPE == 'optimization':
        poisoned_subset, injected_regions = inject_optimization_trigger(data_to_poison)
    else:
        poisoned_subset = data_to_poison
        injected_regions = [(0, data_to_poison.shape[2], 0, data_to_poison.shape[3])] * n_poison
    
    train_dataset.data[poison_indices] = poisoned_subset
    train_dataset.labels[poison_indices] = config.TARGET_CLASS
    train_dataset.set_poison_flags(poison_indices)
    
    all_true_regions = [None] * n_samples
    for idx, poison_idx in enumerate(poison_indices):
        all_true_regions[poison_idx] = injected_regions[idx]
    train_dataset.true_regions = all_true_regions
    
    print(f"Poisoned {n_poison} samples ({config.POISON_RATE*100}%) with {config.TRIGGER_TYPE} trigger.")
    
    try:
        from visualize import plot_poisoned_skeleton
        if n_poison > 0 and config.TRIGGER_TYPE == 'PSBA':
            for img_num in range(min(5, n_poison)): 
                sample_skeleton = poisoned_subset[img_num, :, 32, :, 0] 
                
                trigger_joints = []
                if config.DATASET == 'UV_HUMAN':
                    if config.PSBA_TRIGGER == 'crossing_hands': trigger_joints = [7, 8, 9, 10] 
                    elif config.PSBA_TRIGGER == 'nodding': trigger_joints = [0, 1, 2, 3, 4]
                    elif config.PSBA_TRIGGER == 'bending': trigger_joints = [5, 6, 11, 12]
                    elif config.PSBA_TRIGGER == 'kicking': trigger_joints = [13, 14, 15, 16]
                else: 
                    if config.PSBA_TRIGGER == 'crossing_hands': trigger_joints = [10, 11, 12, 13, 14, 15] 
                    elif config.PSBA_TRIGGER == 'nodding': trigger_joints = [0, 1, 2, 3]
                    elif config.PSBA_TRIGGER == 'bending': trigger_joints = [4, 5, 6, 7, 8, 9]
                    elif config.PSBA_TRIGGER == 'kicking': trigger_joints = [16, 17, 18, 19, 20, 21]

                mode_name = f"PSBA_{config.PSBA_TRIGGER}_{'test' if is_test else 'train'}_sample_{img_num}"

                plot_poisoned_skeleton(
                    skeleton_data=sample_skeleton,
                    trigger_joints=trigger_joints,
                    dataset_name=config.DATASET,
                    epoch="Pre-Train",
                    phase="poison_generation",
                    train_mode=mode_name
                )
    except Exception as e:
        print(f"[Visualization Warning] Failed to plot skeleton heatmap: {e}")
    
    return train_dataset, poison_indices