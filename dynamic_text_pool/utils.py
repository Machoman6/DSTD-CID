"""
Utility functions for dynamic text pool system.
"""

import time
from typing import Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def time_function(func_name: str = ""):
    """Decorator to time function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            name = func_name or func.__name__
            print(f"{name} took {end_time - start_time:.2f} seconds")
            return result
        return wrapper
    return decorator


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory usage information."""
    import torch
    if not torch.cuda.is_available():
        return {"available": False}
    
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
    allocated_memory = torch.cuda.memory_allocated(device) / 1024**3
    cached_memory = torch.cuda.memory_reserved(device) / 1024**3
    free_memory = total_memory - allocated_memory
    
    return {
        "available": True,
        "total_gb": total_memory,
        "allocated_gb": allocated_memory,
        "cached_gb": cached_memory,
        "free_gb": free_memory,
        "utilization": allocated_memory / total_memory * 100
    }


def print_gpu_status():
    """Print current GPU memory status."""
    info = get_gpu_memory_info()
    if info["available"]:
        print(f"GPU Memory - Total: {info['total_gb']:.1f}GB, "
              f"Used: {info['allocated_gb']:.1f}GB ({info['utilization']:.1f}%), "
              f"Free: {info['free_gb']:.1f}GB")
    else:
        print("GPU not available")


def gpu_cosine_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """Compute cosine similarity using GPU acceleration when available."""
    import torch
    
    if not torch.cuda.is_available() or embeddings1.size == 0 or embeddings2.size == 0:
        # Fallback to CPU computation
        return cosine_similarity(embeddings1, embeddings2)
    
    try:
        # Convert to GPU tensors
        device = torch.device('cuda')
        tensor1 = torch.from_numpy(embeddings1).float().to(device)
        tensor2 = torch.from_numpy(embeddings2).float().to(device)
        
        # Normalize tensors
        tensor1 = torch.nn.functional.normalize(tensor1, p=2, dim=1)
        tensor2 = torch.nn.functional.normalize(tensor2, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.mm(tensor1, tensor2.t())
        
        # Convert back to numpy
        result = similarity.cpu().numpy()
        
        # Clean up GPU memory
        del tensor1, tensor2, similarity
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        print(f"GPU cosine similarity failed, falling back to CPU: {e}")
        return cosine_similarity(embeddings1, embeddings2)


