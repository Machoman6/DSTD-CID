"""
Embedding computation functions for dynamic text pool system.
"""

import os
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from tqdm import tqdm


def load_model(model_path: str) -> SentenceTransformer:
    """Load SentenceTransformer model with optimized CUDA settings."""
    # Optimize CUDA memory allocation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")  # Async execution
    
    # Check CUDA availability
    import torch
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        try:
            # Clear GPU cache before loading
            torch.cuda.empty_cache()
            
            # Load model on GPU with optimized settings
            model = SentenceTransformer(
                model_path, 
                device="cuda",
                cache_folder=None  # Use default cache
            )
            
            # Set model to evaluation mode for inference
            model.eval()
            
            # Print GPU memory info
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"Model loaded on GPU. Available GPU memory: {gpu_memory:.1f} GB")
            
            return model
            
        except Exception as e:
            print(f"Failed to load model on GPU: {e}")
            print("Falling back to CPU...")
            torch.cuda.empty_cache()
    
    # Fallback to CPU
    model = SentenceTransformer(model_path, device="cpu")
    model.eval()
    print("Model loaded on CPU")
    return model


def compute_embeddings(model: SentenceTransformer, texts: List[str], batch_size: int = 8, 
                      desc: str = "Computing embeddings") -> np.ndarray:
    """Compute embeddings with CUDA acceleration and memory optimization."""
    if not texts:
        # Determine embedding dimension dynamically to avoid hardcoding
        try:
            dim = model.get_sentence_embedding_dimension()
        except Exception:
            dim = 1
        return np.empty((0, dim))
    
    # Use larger batch size for GPU acceleration
    device = next(model.parameters()).device
    if device.type == 'cuda':
        # Use a smaller, safer batch size on large embedding models
        batch_size = min(batch_size, 8)
        # Clear GPU cache before processing
        import torch
        torch.cuda.empty_cache()
    
    # Calculate number of batches for progress bar
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    # Create progress bar
    pbar = tqdm(total=num_batches, desc=desc, unit="batch", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    try:
        # Process in batches with progress tracking
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_emb = model.encode(batch_texts, batch_size=len(batch_texts), 
                                   show_progress_bar=False, convert_to_numpy=True)
            all_embeddings.append(batch_emb)
            pbar.update(1)
            pbar.set_postfix({
                'batch': f'{i//batch_size + 1}/{num_batches}',
                'texts': len(batch_texts)
            })
        
        # Concatenate all embeddings
        emb = np.vstack(all_embeddings)
        emb = normalize(emb, norm="l2")
        
    finally:
        pbar.close()
    
    # Clear GPU cache after processing
    if device.type == 'cuda':
        import torch
        torch.cuda.empty_cache()
    
    return emb

