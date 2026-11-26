"""
Complete SCARF Model Implementations - Fixed Version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import config
from utils import compute_relative_transfer, apply_temporal_smoothing, validate_manifold_constraints

# ============================================
# 1. ST-GCN Backbone (Victim Model) - Simplified
# ============================================

class GraphConv(nn.Module):
    """Graph convolution layer for ST-GCN"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(1, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x, A):
        N, C, T, V = x.shape
        x = self.conv(x)
        x = x.view(N, self.kernel_size, self.out_channels, T, V)
        x = torch.einsum('nkctv,vu->nkctu', x, A)
        x = x.mean(dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TemporalConv(nn.Module):
    """Temporal convolution layer"""
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=((kernel_size-1)//2, 0),
            stride=(stride, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class STGCNBlock(nn.Module):
    """Spatial-Temporal Graph Convolution Block with Dropout"""
    def __init__(self, in_channels, out_channels, temporal_kernel=5, stride=1, dropout=0.5):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels)
        self.tcn = TemporalConv(out_channels, out_channels, temporal_kernel, stride)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, A):
        x = self.gcn(x, A)
        x = self.tcn(x)
        x = self.dropout(x)
        return x

class STGCN(nn.Module):
    """Spatial Temporal Graph Convolution Network"""
    def __init__(self, num_classes=None, in_channels=None, num_joints=None):
        super().__init__()
        self.num_classes = num_classes if num_classes is not None else config.NUM_CLASSES
        self.num_joints = num_joints if num_joints is not None else config.NUM_JOINTS
        self.in_channels = in_channels if in_channels is not None else config.INPUT_CHANNELS
        
        self.register_buffer('A', self._create_adj_matrix(self.num_joints))
        self.embed = nn.Conv2d(self.in_channels, 64, 1)
        
        self.block1 = STGCNBlock(64, 64, temporal_kernel=5, stride=1)
        self.block2 = STGCNBlock(64, 128, temporal_kernel=5, stride=2)
        self.block3 = STGCNBlock(128, 128, temporal_kernel=5, stride=1)
        self.block4 = STGCNBlock(128, 256, temporal_kernel=5, stride=2)
        self.block5 = STGCNBlock(256, 256, temporal_kernel=5, stride=1)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, self.num_classes)
        self.intermediate_features = None
        
    def _create_adj_matrix(self, num_joints):
        A = torch.zeros((num_joints, num_joints))
        if num_joints == 17:
            edges = [
                (0, 1), (0, 2), (1, 3), (2, 4),
                (0, 5), (0, 6), (5, 7), (7, 9),
                (6, 8), (8, 10), (5, 11), (6, 12),
                (11, 13), (13, 15), (12, 14), (14, 16)
            ]
        else:
            edges = [
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                (22, 23), (23, 8), (24, 25), (25, 12)
            ]
            edges = [(i - 1, j - 1) for (i, j) in edges]

        for i, j in edges:
            A[i, j] = 1
            A[j, i] = 1
        for i in range(num_joints):
            A[i, i] = 1
            
        d = A.sum(1)
        d_inv = torch.pow(d, -0.5)
        d_inv[torch.isinf(d_inv)] = 0.0
        D_inv = torch.diag(d_inv)
        A_norm = torch.mm(torch.mm(D_inv, A), D_inv)
        return A_norm
    
    def forward(self, x, return_features=False):
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(N * M, C, T, V)
        x = self.embed(x)
        
        x = self.block1(x, self.A)
        x = self.block2(x, self.A)
        x = self.block3(x, self.A)
        x = self.block4(x, self.A)
        x = self.block5(x, self.A)
        
        if return_features:
            self.intermediate_features = x.detach().cpu().numpy()
            
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = x.view(N, M, -1).mean(dim=1)
        x = self.fc(x)
        return x

# ============================================
# 2. Completion Model
# ============================================

class CompletionModel(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(hidden_dim, in_channels, kernel_size=3, padding=1),
        )
        
    def forward(self, x, mask_region=None):
        # This model is trained on the person-averaged (C,T,V) representation
        # (see train_completion.py: data.mean(dim=-1)). It is CORRECT for this
        # model to keep collapsing the M axis internally -- that matches what
        # it saw during training. Callers must restore scale (x2.0) and only
        # write the output back into actor 0's slot of the full (C,T,V,M) tensor.
        if len(x.shape) == 5:
            x = x[..., 0]
        x = x.unsqueeze(-1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.squeeze(-1)
        return decoded, mask_region

# ============================================
# 3. SCARF Detector Components
# ============================================

class RegionProposalNetwork:
    """Proposes candidate node-time regions using gradient-based saliency"""
    def __init__(self, num_joints=25, num_frames=64, window_sizes=[5, 10, 15]):
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.window_sizes = window_sizes
        
    def propose(self, model, x, target_class=None):
        # x is (C, T, V, M) -- the FULL shape the victim model was trained on.
        # Do NOT unsqueeze(-1) here; the person axis M already exists.
        x_input = x.clone().detach().unsqueeze(0).requires_grad_(True)  # (1, C, T, V, M)
        output = model(x_input)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        target_logit = output[0, target_class]
        target_logit.backward()

        saliency = x_input.grad.abs().mean(dim=(1, 4)).squeeze(0).cpu().numpy()
        
        regions = []
        scores = []
        for w_size in self.window_sizes:
            max_score = -1
            best_region = None
            for t in range(0, self.num_frames - w_size, w_size // 2):
                for v in range(0, self.num_joints - w_size, w_size // 2):
                    region_score = saliency[t:t+w_size, v:v+w_size].mean()
                    if region_score > max_score:
                        max_score = region_score
                        best_region = (t, t+w_size, v, v+w_size)
            if best_region and max_score > 0.01:
                regions.append(best_region)
                scores.append(max_score)
                
        if len(regions) > 5:
            indices = np.argsort(scores)[-5:][::-1]
            regions = [regions[i] for i in indices]
        return regions

def calculate_margin(logits, target_class):
    """Computes M_k(X) = z_k(X) - max_{j != k} z_j(X)"""
    z_k = logits[0, target_class].item()
    mask = torch.ones_like(logits[0], dtype=torch.bool)
    mask[target_class] = False
    max_j = logits[0][mask].max().item()
    return z_k - max_j

class LocalizedInfluenceScorer:
    """Computes localized influence using explicit counterfactual masking"""
    def __init__(self, completion_model, consensus_predictor=None):
        self.completion = completion_model
        self.consensus = consensus_predictor
        
    def compute(self, model, x, region):
        # x is (C, T, V, M) -- full shape, both actors present.
        t_start, t_end, v_start, v_end = region
        x_input = x.clone().detach().unsqueeze(0)  # (1, C, T, V, M)
        
        # 1. Get original logits
        with torch.no_grad():
            orig_logits = model(x_input)
            orig_pred = orig_logits.argmax(dim=1).item()
            
        # 2. Explicit Masking & Reconstruction -- mask ONLY the primary actor (M=0)
        masked_x = x.clone()
        masked_x[:, t_start:t_end, v_start:v_end, 0] = 0.0
        
        with torch.no_grad():
            masked_in = masked_x.unsqueeze(0)          # (1, C, T, V, M)
            recon, _ = self.completion(masked_in)
            recon = recon.squeeze(0)                    # (C, T, V)
            # CompletionModel was trained on mean(dim=-1) targets. Since the
            # second person slot is always zero, that mean is exactly half of
            # actor 0's true scale -- restore it before writing back.
            # recon = recon * 2.0
            
            # Counterfactual sample X^r: replace ONLY the candidate region of actor 0
            counterfactual_x = x.clone()
            counterfactual_x[:, t_start:t_end, v_start:v_end, 0] = recon[:, t_start:t_end, v_start:v_end]
            recon_input = counterfactual_x.unsqueeze(0)  # (1, C, T, V, M)
            
        # 3. Get prediction on counterfactual
        with torch.no_grad():
            recon_logits = model(recon_input)
            recon_pred = recon_logits.argmax(dim=1).item()
            
        # 4. Calculate raw logit change: Delta(r) = z(X) - z(X^r)
        delta_r = orig_logits - recon_logits
        
        # 5. Identify the affected class
        k_star = delta_r[0].argmax().item()
        
        # 6. Compute Pre- and Post-Intervention Margins
        margin_orig = calculate_margin(orig_logits, k_star)
        margin_recon = calculate_margin(recon_logits, k_star)
        
        # 7. Calculate Affected-Class Margin Change: DeltaM(r)
        delta_margin = margin_orig - margin_recon
        logit_k_star_change = delta_r[0, k_star].item()
        logit_magnitude = torch.norm(delta_r).item()
        
        if self.consensus:
            label_preserved = self.consensus.check(model, x, counterfactual_x, orig_pred)
        else:
            label_preserved = (recon_pred == orig_pred)
            
        return {
            'k_star': k_star,
            'delta_margin': delta_margin,
            'delta_logit': logit_k_star_change,
            'magnitude': logit_magnitude,
            'orig_pred': orig_pred,
            'recon_pred': recon_pred,
            'label_preserved': label_preserved
        }

class SignatureExtractor:
    """Extracts normalized signatures from regions for clustering"""
    def __init__(self, target_length=256):
        self.target_length = target_length
        
    def extract(self, x, region):
        # x is (C, T, V, M) -- extract purely from the primary actor (M=0)
        t_start, t_end, v_start, v_end = region
        region_data = x[:, t_start:t_end, v_start:v_end, 0].cpu().numpy()
        region_data = region_data + np.random.randn(*region_data.shape) * 0.01
        
        if v_start <= 0 <= v_end-1:
            root_idx = 0 - v_start
            if 0 <= root_idx < region_data.shape[2]:
                root = region_data[:, :, root_idx:root_idx+1]
                region_data = region_data - root
                
        velocities = np.diff(region_data, axis=1)
        velocities = np.pad(velocities, ((0,0), (0,1), (0,0)), mode='constant')
        signature = np.concatenate([region_data.flatten(), velocities.flatten()])
        
        if len(signature) > self.target_length:
            indices = np.linspace(0, len(signature) - 1, self.target_length).astype(int)
            signature = signature[indices]
        elif len(signature) < self.target_length:
            pad_width = self.target_length - len(signature)
            signature = np.pad(signature, (0, pad_width), mode='constant')
            
        norm = np.linalg.norm(signature) + 1e-8
        signature = signature / norm
        return signature

# class TransferEvaluator:
#     """Evaluates cross-context transferability"""
#     def __init__(self):
#         pass
        
#     def evaluate(self, model, source_x, region, k_star, reference_samples):
#         # source_x and each ref_x are (C, T, V, M) -- work on actor 0 only.
#         t_start, t_end, v_start, v_end = region
#         margin_increases = []
#         induced_predictions = 0
#         valid_transfers = 0
        
#         for ref_x in reference_samples:
#             ref_input = ref_x.unsqueeze(0)  # (1, C, T, V, M) -- no fake unsqueeze(-1)
#             with torch.no_grad():
#                 base_logits = model(ref_input)
#                 base_margin = calculate_margin(base_logits, k_star)
                
#                 transferred_region = compute_relative_transfer(
#                     source_x[:, :, :, 0], ref_x[:, :, :, 0], region
#                 )
#                 transferred_x = ref_x.clone()
#                 transferred_x[:, t_start:t_end, v_start:v_end, 0] = transferred_region
#                 transferred_x[:, :, :, 0] = apply_temporal_smoothing(
#                     transferred_x[:, :, :, 0], t_start, t_end
#                 )
                
#                 if not validate_manifold_constraints(ref_x[:, :, :, 0], transferred_x[:, :, :, 0]):
#                     continue
                    
#                 valid_transfers += 1
#                 transferred_input = transferred_x.unsqueeze(0)  # (1, C, T, V, M)
#                 transfer_logits = model(transferred_input)
#                 transfer_margin = calculate_margin(transfer_logits, k_star)
                
#                 margin_increases.append(transfer_margin - base_margin)
#                 if transfer_logits.argmax(dim=1).item() == k_star:
#                     induced_predictions += 1
                    
#         transfer_margin_score = sum(margin_increases) / valid_transfers if valid_transfers > 0 else 0.0
#         prediction_rate = induced_predictions / valid_transfers if valid_transfers > 0 else 0.0
        
#         return {
#             'transfer_margin': transfer_margin_score,
#             'prediction_rate': prediction_rate,
#             'valid_transfers': valid_transfers
#         }

# class ConsensusPredictor:
#     """Ensures label preservation is robust using multiple predictions"""
#     def __init__(self, num_seeds=3, noise_level=0.01):
#         self.num_seeds = num_seeds
#         self.noise_level = noise_level
        
#     def check(self, model, x, x_recon, target_class):
#         # x and x_recon now arrive as (C, T, V, M) -- unsqueeze(0) is sufficient.
#         device = next(model.parameters()).device
#         x = x.to(device)
#         predictions = []
        
#         with torch.no_grad():
#             x_input = x_recon.unsqueeze(0)  # (1, C, T, V, M)
#             recon_output = model(x_input)
#             recon_pred = recon_output.argmax(dim=1).item()
#             predictions.append(recon_pred)
            
#         for _ in range(self.num_seeds):
#             noise = torch.randn_like(x) * self.noise_level
#             noisy_x = x + noise
#             with torch.no_grad():
#                 noisy_input = noisy_x.unsqueeze(0)  # (1, C, T, V, M)
#                 output = model(noisy_input)
#                 pred = output.argmax(dim=1).item()
#                 predictions.append(pred)
                
#         from collections import Counter
#         counter = Counter(predictions)
#         most_common_pred, count = counter.most_common(1)[0]
#         return (most_common_pred == target_class) and (count >= len(predictions) * 0.6)


class ConsensusPredictor:
    """Optimized: Checks all seeds in a SINGLE batched forward pass"""
    def __init__(self, num_seeds=3, noise_level=0.01):
        self.num_seeds = num_seeds
        self.noise_level = noise_level
        
    def check(self, model, x, x_recon, target_class):
        device = next(model.parameters()).device
        x = x.to(device)
        
        # Combine all variants into ONE batch
        batch = [x_recon]
        for _ in range(self.num_seeds):
            batch.append(x + torch.randn_like(x) * self.noise_level)
            
        batch_tensor = torch.stack(batch) # Shape: (4, C, T, V, M)
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            preds = outputs.argmax(dim=1).tolist()
            
        from collections import Counter
        counter = Counter(preds)
        most_common_pred, count = counter.most_common(1)[0]
        return (most_common_pred == target_class) and (count >= len(preds) * 0.6)

class TransferEvaluator:
    """Optimized: Calculates cross-context transfer in ONE batched forward pass"""
    def __init__(self):
        self.base_margins_cache = {}
        
    def evaluate(self, model, source_x, region, k_star, reference_samples):
        t_start, t_end, v_start, v_end = region
        
        # 1. Precompute base logits in ONE batch (Runs only once per k_star)
        if k_star not in self.base_margins_cache:
            ref_batch = torch.stack(reference_samples) 
            with torch.no_grad():
                base_logits = model(ref_batch)
            self.base_margins_cache[k_star] = [calculate_margin(base_logits[i:i+1], k_star) for i in range(len(reference_samples))]
            
        base_margins = self.base_margins_cache[k_star]
        
        # 2. Prepare transferred samples
        transferred_list = []
        valid_indices = []
        
        for i, ref_x in enumerate(reference_samples):
            transferred_region = compute_relative_transfer(source_x[:, :, :, 0], ref_x[:, :, :, 0], region)
            transferred_x = ref_x.clone()
            transferred_x[:, t_start:t_end, v_start:v_end, 0] = transferred_region
            transferred_x[:, :, :, 0] = apply_temporal_smoothing(transferred_x[:, :, :, 0], t_start, t_end)
            
            if validate_manifold_constraints(ref_x[:, :, :, 0], transferred_x[:, :, :, 0]):
                transferred_list.append(transferred_x)
                valid_indices.append(i)
                
        if not transferred_list:
            return {'transfer_margin': 0.0, 'prediction_rate': 0.0, 'valid_transfers': 0}
            
        # 3. BATCH process all transferred samples in ONE forward pass
        trans_batch = torch.stack(transferred_list)
        with torch.no_grad():
            transfer_logits = model(trans_batch)
            
        margin_increases = []
        induced_predictions = 0
        
        for idx in range(len(transferred_list)):
            orig_idx = valid_indices[idx]
            t_margin = calculate_margin(transfer_logits[idx:idx+1], k_star)
            margin_increases.append(t_margin - base_margins[orig_idx])
            
            if transfer_logits[idx].argmax(dim=0).item() == k_star:
                induced_predictions += 1
                
        valid_transfers = len(transferred_list)
        return {
            'transfer_margin': sum(margin_increases) / valid_transfers,
            'prediction_rate': induced_predictions / valid_transfers,
            'valid_transfers': valid_transfers
        }

class SCARFDetector:
    """Complete SCARF detection pipeline using Logit-Margin Repulsion"""
    def __init__(self, victim_model, completion_model, config):
        self.victim = victim_model
        self.completion = completion_model
        self.config = config
        
        self.region_proposal = RegionProposalNetwork(config.NUM_JOINTS, config.NUM_FRAMES, config.WINDOW_SIZES)
        self.consensus = ConsensusPredictor(config.CONSENSUS_SEEDS)
        
        self.influence_scorer = LocalizedInfluenceScorer(self.completion, self.consensus)
        self.transfer_evaluator = TransferEvaluator()
        self.signature_extractor = SignatureExtractor()
        
        self.scores = []
        self.regions = []
        self.signatures = []
        
    def detect(self, data_loader, reference_pool=None):
        all_scores = []
        all_regions = []
        all_signatures = []
        
        self.victim.eval()
        self.completion.eval()
        
        for batch_idx, (x, labels, poison_flags) in enumerate(data_loader):
            x = x.to(self.config.DEVICE)
            
            for i in range(len(x)):
                # Keep the FULL (C, T, V, M) shape -- this is what the victim
                # model was trained on. Do NOT collapse the person axis here.
                sample = x[i]
                
                regions = self.region_proposal.propose(self.victim, sample)
                
                if not regions:
                    all_scores.append(0.0)
                    all_regions.append(None)
                    all_signatures.append(np.zeros(self.signature_extractor.target_length))
                    continue
                
                region_influences = []
                region_transfers = []
                region_data = []
                
                for region in regions:
                    influence_metrics = self.influence_scorer.compute(self.victim, sample, region)
                    
                    if reference_pool:
                        transfer_metrics = self.transfer_evaluator.evaluate(
                            self.victim, sample, region, 
                            influence_metrics['k_star'], reference_pool
                        )
                    else:
                        transfer_metrics = {'transfer_margin': 0.0, 'prediction_rate': 0.0}
                        
                    sig = self.signature_extractor.extract(sample, region)
                    region_influences.append(influence_metrics['delta_margin'])
                    region_transfers.append(transfer_metrics['transfer_margin'])
                    region_data.append((region, sig))
                
                # FIX: Combine absolute margins directly. No per-sample normalization.
                scores = [max(0.0, inf) + max(0.0, tr) for inf, tr in zip(region_influences, region_transfers)]
                best_idx = int(np.argmax(scores))
                
                all_scores.append(scores[best_idx])
                all_regions.append(region_data[best_idx][0])
                all_signatures.append(region_data[best_idx][1])
            
            if batch_idx % 10 == 0:
                print(f"  Processed {batch_idx}/{len(data_loader)} batches")
        
        all_scores = np.array(all_scores)
        threshold = np.percentile(all_scores, (1 - self.config.POISON_RATE) * 100)
        poison_flags = all_scores >= threshold
        
        self.scores = all_scores
        self.regions = all_regions
        self.signatures = np.array(all_signatures)
        return all_scores, poison_flags, all_regions
        
    def cluster_signatures(self):
        """Cluster signatures to find recurring patterns"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        if len(self.signatures) < 10:
            return None
        
        signatures = self.signatures + np.random.randn(*self.signatures.shape) * 0.001
        best_score = -1
        best_labels = None
        
        for k in range(2, min(5, len(signatures) // 5)):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(signatures)
                if len(set(labels)) > 1:
                    score = silhouette_score(signatures, labels)
                    if score > best_score:
                        best_score = score
                        best_labels = labels
            except:
                continue
        
        if best_labels is not None:
            print(f"  Found {len(set(best_labels))} clusters (silhouette: {best_score:.3f})")
        return best_labels