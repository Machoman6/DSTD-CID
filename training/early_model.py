#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight early prediction model:
- Text time series encoder: Transformer over hour-level pooled embeddings
- Numeric time series encoder: Transformer over [malicious,total] sequences
- Fusion: concat -> MLP -> p(h) per hour
- Joint loss: Lclass + λ1·Luncertainty + λ2·Ltrend  (delay penalty removed)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, L, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return x + self.pe[:, :L]


class SeriesTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4, nlayers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pos = PositionalEncoding(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, L, d_in)
        h = self.proj(x)
        h = self.pos(h)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return h  # (B, L, d_model)


class SeriesDecomp(nn.Module):
    """Simple moving average decomposition used in Autoformer family: x = trend + seasonal."""
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        # depthwise conv as moving average
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.register_buffer(
            'kernel', torch.ones(1, 1, kernel_size) / kernel_size
        )

    def moving_average(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d) -> reshape to (B*d, 1, L) for conv
        B, L, D = x.shape
        y = x.transpose(1, 2).reshape(B * D, 1, L)
        trend = torch.conv1d(y, self.kernel, padding=self.padding)
        trend = trend.reshape(B, D, L).transpose(1, 2)
        return trend

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_average(x)
        seasonal = x - trend
        return seasonal, trend


class AutoformerEncoderLite(nn.Module):
    """Lightweight Autoformer-style encoder: decompose -> encode seasonal and trend -> fuse.
    This is a minimal in-project implementation to avoid external dependencies.
    """
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4, nlayers: int = 2, dropout: float = 0.1, kernel_size: int = 25):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        self.decomp = SeriesDecomp(kernel_size=kernel_size)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.seasonal_enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.trend_enc = nn.TransformerEncoder(enc_layer, num_layers=max(1, nlayers // 2))
        self.fuse = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, L, d_in)
        h = self.proj(x)
        h = self.pos(h)
        seasonal, trend = self.decomp(h)
        hs = self.seasonal_enc(seasonal, src_key_padding_mask=key_padding_mask)
        ht = self.trend_enc(trend, src_key_padding_mask=key_padding_mask)
        out = torch.cat([hs, ht], dim=-1)
        out = self.fuse(out)
        return out  # (B, L, d_model)


class ECTSModel(nn.Module):
    def __init__(self, text_dim: int = 256, num_dim: int = 2, d_model: int = 128):
        super().__init__()
        self.text_enc = SeriesTransformer(input_dim=text_dim, d_model=d_model)
        self.num_enc = AutoformerEncoderLite(input_dim=num_dim, d_model=d_model)
        
        # Multi-head attention mechanism for feature fusion
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
        # Projection layer after feature fusion
        self.fusion_projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classification head: predict class
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(128, 1)
        )
        
        # Stop head: predict whether to stop observing
        self.stopper = nn.Sequential(
            nn.Linear(d_model, 128), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights for numerical stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, text_seq: torch.Tensor, num_seq: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # text_seq: (B, 1, dt), num_seq: (B, L, 2)
        # Text sequence has only 1 time step, numeric sequence has L time steps
        
        # Process text sequence
        ht = self.text_enc(text_seq)  # (B, 1, d) - no padding mask needed, only 1 time step
        
        # Process numeric sequence
        hn = self.num_enc(num_seq, key_padding_mask)    # (B, L, d)
        
        # Broadcast text features to all time steps
        ht_expanded = ht.expand(-1, hn.size(1), -1)     # (B, L, d)
        
        # Use multi-head attention for feature fusion
        # Query: numeric sequence features, Key/Value: text features
        # This allows each time step of numeric sequence to attend to text features
        attn_output, attn_weights = self.multihead_attention(
            query=hn,           # (B, L, d) - numeric sequence as query
            key=ht_expanded,    # (B, L, d) - text features as key
            value=ht_expanded,  # (B, L, d) - text features as value
            key_padding_mask=key_padding_mask
        )
        
        # Concatenate attention output with original numeric features
        h_concat = torch.cat([attn_output, hn], dim=-1)  # (B, L, 2d)
        
        # Feature fusion through projection layer
        h = self.fusion_projection(h_concat)             # (B, L, d)
        
        # Classification prediction - output logits directly for BCEWithLogits
        class_logits = self.classifier(h).squeeze(-1)   # (B, L)
        
        # Stop prediction - output logits directly for BCEWithLogits
        stop_logits = self.stopper(h).squeeze(-1)       # (B, L)
        
        return class_logits, stop_logits


class ECTSModelStopNumOnly(nn.Module):
    """
    Ablation: Stop head only uses numeric features.
    This helps verify if dynamic text pool helps model make earlier stop decisions.
    """
    def __init__(self, text_dim: int = 256, num_dim: int = 2, d_model: int = 128):
        super().__init__()
        self.text_enc = SeriesTransformer(input_dim=text_dim, d_model=d_model)
        self.num_enc = AutoformerEncoderLite(input_dim=num_dim, d_model=d_model)
        
        # Multi-head attention for feature fusion
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
        # Feature fusion projection
        self.fusion_projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classifier: uses fused features
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(128, 1)
        )
        
        # Stop head: ONLY uses numeric features (ablation)
        self.stopper = nn.Sequential(
            nn.Linear(d_model, 128), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(128, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights for numerical stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, text_seq: torch.Tensor, num_seq: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # text_seq: (B, 1, dt), num_seq: (B, L, 2)
        
        # Process text sequence
        ht = self.text_enc(text_seq)  # (B, 1, d)
        
        # Process numeric sequence
        hn = self.num_enc(num_seq, key_padding_mask)    # (B, L, d)
        
        # Broadcast text features to all time steps
        ht_expanded = ht.expand(-1, hn.size(1), -1)     # (B, L, d)
        
        # Multi-head attention fusion
        attn_output, attn_weights = self.multihead_attention(
            query=hn,
            key=ht_expanded,
            value=ht_expanded,
            key_padding_mask=key_padding_mask
        )
        
        # Concatenate attention output with original numeric features
        h_concat = torch.cat([attn_output, hn], dim=-1)  # (B, L, 2d)
        
        # Feature fusion projection
        h = self.fusion_projection(h_concat)             # (B, L, d)
        
        # Classification: uses fused features
        class_logits = self.classifier(h).squeeze(-1)   # (B, L)
        
        # Stop prediction: ONLY uses numeric features (ablation)
        stop_logits = self.stopper(hn).squeeze(-1)       # (B, L) - only hn, not h
        
        return class_logits, stop_logits


class ECTSModelStopTextOnly(nn.Module):
    """
    Ablation: Stop head only uses text features.
    This helps verify if dynamic text pool helps model make earlier stop decisions.
    """
    def __init__(self, text_dim: int = 256, num_dim: int = 2, d_model: int = 128):
        super().__init__()
        self.text_enc = SeriesTransformer(input_dim=text_dim, d_model=d_model)
        self.num_enc = AutoformerEncoderLite(input_dim=num_dim, d_model=d_model)
        
        # Multi-head attention for feature fusion
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
        # Feature fusion projection
        self.fusion_projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classifier: uses fused features
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(128, 1)
        )
        
        # Stop head: ONLY uses text features (ablation)
        self.stopper = nn.Sequential(
            nn.Linear(d_model, 128), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(128, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights for numerical stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, text_seq: torch.Tensor, num_seq: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # text_seq: (B, 1, dt), num_seq: (B, L, 2)
        
        # Process text sequence
        ht = self.text_enc(text_seq)  # (B, 1, d)
        
        # Process numeric sequence
        hn = self.num_enc(num_seq, key_padding_mask)    # (B, L, d)
        
        # Broadcast text features to all time steps
        ht_expanded = ht.expand(-1, hn.size(1), -1)     # (B, L, d)
        
        # Multi-head attention fusion
        attn_output, attn_weights = self.multihead_attention(
            query=hn,
            key=ht_expanded,
            value=ht_expanded,
            key_padding_mask=key_padding_mask
        )
        
        # Concatenate attention output with original numeric features
        h_concat = torch.cat([attn_output, hn], dim=-1)  # (B, L, 2d)
        
        # Feature fusion projection
        h = self.fusion_projection(h_concat)             # (B, L, d)
        
        # Classification: uses fused features
        class_logits = self.classifier(h).squeeze(-1)   # (B, L)
        
        # Stop prediction: ONLY uses text features (ablation)
        stop_logits = self.stopper(ht_expanded).squeeze(-1)  # (B, L) - only ht_expanded, not h
        
        return class_logits, stop_logits


@dataclass
class LossWeights:
    lambda_early: float = 1.0
    lambda_uncert: float = 0.5
    lambda_trend: float = 0.3


def trend_consistency_loss(class_logits: torch.Tensor, mal_seq: torch.Tensor, 
                          key_padding_mask: torch.Tensor) -> torch.Tensor:
    """
    Trend consistency loss: encourage model's predicted probability trend to match malicious comment count trend
    
    Args:
        class_logits: (B, L) classification logits
        mal_seq: (B, L) malicious comment count sequence
        key_padding_mask: (B, L) padding mask
    
    Returns:
        trend_loss: trend consistency loss
    """
    B, L = class_logits.shape
    
    # Compute classification probabilities
    class_probs = torch.sigmoid(class_logits)  # (B, L)
    
    # Compute trends
    trend_loss = torch.tensor(0.0, device=class_logits.device)
    valid_samples = 0
    
    for i in range(B):
        # Get valid time steps
        valid_mask = ~key_padding_mask[i]  # False indicates valid position
        if valid_mask.sum() < 2:  # At least 2 time steps needed to compute trend
            continue
            
        valid_indices = torch.where(valid_mask)[0]
        if len(valid_indices) < 2:
            continue
            
        # Get probabilities and malicious comment counts for valid time steps
        valid_probs = class_probs[i, valid_indices]  # (T,)
        valid_mal = mal_seq[i, valid_indices]       # (T,)
        
        # Compute trend directions
        # Probability trend: next time step - previous time step
        prob_trend = valid_probs[1:] - valid_probs[:-1]  # (T-1,)
        # Malicious comment trend: next time step - previous time step  
        mal_trend = valid_mal[1:] - valid_mal[:-1]       # (T-1,)
        
        # Compute trend consistency
        # If both trends have same direction (both positive or both negative), loss is 0
        # If trends have opposite directions, loss is incurred
        trend_consistency = prob_trend * mal_trend  # (T-1,)
        
        # Negative trend_consistency indicates opposite trends, needs penalty
        # Use ReLU(-trend_consistency) to penalize opposite trends
        opposite_trend_penalty = torch.relu(-trend_consistency)
        
        # Compute trend loss for this sample
        sample_trend_loss = opposite_trend_penalty.mean()
        trend_loss += sample_trend_loss
        valid_samples += 1
    
    if valid_samples > 0:
        trend_loss = trend_loss / valid_samples
    else:
        trend_loss = torch.tensor(0.0, device=class_logits.device)
    
    return trend_loss


def ects_loss(class_logits: torch.Tensor, stop_logits: torch.Tensor, labels: torch.Tensor, 
              key_padding_mask: torch.Tensor | None, mal_seq: torch.Tensor | None = None,
              alpha: float = 1.0, beta: float = 0.1, gamma: float = 0.5, delta: float = 0.3) -> Dict[str, torch.Tensor]:
    """
    ECTS loss function - compute loss only at the last time step (explicit prefix expansion)
    - class_logits: (B, L) classification logits at each time step
    - stop_logits: (B, L) stop logits at each time step
    - labels: (B,) event labels
    - key_padding_mask: (B, L) True indicates padding position
    - alpha: classification accuracy weight
    - beta: delay penalty weight  
    - gamma: stop decision supervised learning weight
    """
    B, L = class_logits.shape
    mask = ~key_padding_mask if key_padding_mask is not None else torch.ones_like(class_logits, dtype=torch.bool)
    
    # 1. Classification loss: compute only at the last valid time step
    # Find the last valid time step for each sample
    last_valid_indices = torch.zeros(B, dtype=torch.long, device=class_logits.device)
    for i in range(B):
        valid_mask = mask[i]
        if valid_mask.any():
            # Find the last valid position
            last_valid_indices[i] = valid_mask.sum().item() - 1
        else:
            last_valid_indices[i] = 0
    
    # Ensure indices are within bounds
    last_valid_indices = torch.clamp(last_valid_indices, 0, L-1)
    
    # Compute classification loss only at the last time step
    class_logits_last = class_logits[torch.arange(B), last_valid_indices]  # (B,)
    class_logits_clipped = torch.clamp(class_logits_last, min=-10, max=10)
    
    # Check for NaN or Inf
    if not torch.isfinite(class_logits_clipped).all():
        print(f"Warning: Non-finite class_logits detected: {class_logits_clipped}")
        class_logits_clipped = torch.nan_to_num(class_logits_clipped, nan=0.0, posinf=10.0, neginf=-10.0)
    
    class_loss = F.binary_cross_entropy_with_logits(class_logits_clipped, labels.float())
    
    # 2. Stop loss: confidence-based stop decision
    # Compute classification confidence
    class_probs_last = torch.sigmoid(class_logits_clipped)  # (B,)
    
    # Stop decision: intelligent stopping based on confidence and labels
    # Ideal case:
    # - Positive samples: as time progresses, classification probability and stop probability should both increase
    # - Negative samples: as time progresses, classification probability decreases, stop probability increases
    
    positive_samples = labels.bool()  # Positive sample mask
    negative_samples = ~positive_samples  # Negative sample mask
    
    # Initialize stop targets
    stop_targets_last = torch.zeros_like(class_probs_last, dtype=torch.float32)
    
    # Positive samples: stop when confidence is high (class_prob > 0.8)
    if positive_samples.any():
        # For positive samples, stop when model is confident this is cyberbullying
        stop_targets_last[positive_samples] = (class_probs_last[positive_samples] > 0.8).float()
    
    # Negative samples: stop when confidence is low (class_prob < 0.2)
    if negative_samples.any():
        # For negative samples, stop when model is confident this is not cyberbullying
        # i.e., when classification probability is low, model is confident "not cyberbullying"
        stop_targets_last[negative_samples] = (class_probs_last[negative_samples] < 0.2).float()
    
    stop_logits_last = stop_logits[torch.arange(B), last_valid_indices]  # (B,)
    stop_logits_clipped = torch.clamp(stop_logits_last, min=-10, max=10)
    
    # Check for NaN or Inf
    if not torch.isfinite(stop_logits_clipped).all():
        print(f"Warning: Non-finite stop_logits detected: {stop_logits_clipped}")
        stop_logits_clipped = torch.nan_to_num(stop_logits_clipped, nan=0.0, posinf=10.0, neginf=-10.0)
    
    stop_loss = F.binary_cross_entropy_with_logits(stop_logits_clipped, stop_targets_last)
    
    # 3. Delay penalty: removed, always 0
    delay_penalty = torch.tensor(0.0, device=class_logits.device)
    
    # 4. Trend consistency loss: encourage predicted probability trend to match malicious comment count trend
    trend_loss = torch.tensor(0.0, device=class_logits.device)
    if mal_seq is not None:
        trend_loss = trend_consistency_loss(class_logits, mal_seq, key_padding_mask)
    
    # Check if all loss components are finite
    if not torch.isfinite(class_loss):
        print(f"Warning: Non-finite class_loss: {class_loss}")
        class_loss = torch.tensor(0.0, device=class_logits.device)
    
    if not torch.isfinite(stop_loss):
        print(f"Warning: Non-finite stop_loss: {stop_loss}")
        stop_loss = torch.tensor(0.0, device=class_logits.device)
    
    if not torch.isfinite(trend_loss):
        print(f"Warning: Non-finite trend_loss: {trend_loss}")
        trend_loss = torch.tensor(0.0, device=class_logits.device)
    
    # 5. Total loss: includes trend consistency loss
    total_loss = alpha * class_loss + beta * delay_penalty + gamma * stop_loss + delta * trend_loss
    
    # Final check for total loss
    if not torch.isfinite(total_loss):
        print(f"Warning: Non-finite total_loss: {total_loss}")
        total_loss = torch.tensor(0.0, device=class_logits.device)
    
    return {
        "total": total_loss,
        "class": class_loss.detach(),
        "delay": delay_penalty.detach(),
        "stop_loss": stop_loss.detach(),
        "trend": trend_loss.detach(),
    }


