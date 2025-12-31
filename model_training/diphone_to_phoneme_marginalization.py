"""
Diphone to Phoneme Marginalization

This module provides functions to convert diphone logits/probabilities to phoneme logits/probabilities
by marginalizing over all possible previous phonemes.

Diphone ID formula: diphone_id = (prev - 1) * P + (next - 1) + 1
where:
- prev: previous phoneme ID (1-indexed, 1 = first phoneme after BLANK)
- next: next phoneme ID (1-indexed, 1 = first phoneme after BLANK)
- P: number of non-blank phoneme classes (mono_n_classes - 1)

To marginalize diphone probabilities to phoneme probabilities:
- For phoneme ph (1-indexed), sum all diphone probabilities where next = ph
- Diphone IDs ending with ph: [(prev - 1) * P + (ph - 1) + 1 for prev in range(1, P+1)]
"""

import torch
import numpy as np
from typing import Union


def marginalize_diphone_logits_to_phoneme_logits(
    diphone_logits: Union[torch.Tensor, np.ndarray],
    mono_n_classes: int,
    use_log_space: bool = True
) -> Union[torch.Tensor, np.ndarray]:
    """
    Marginalize diphone logits to phoneme logits by summing probabilities of all diphones
    that end with the same phoneme.
    
    Args:
        diphone_logits: Array of shape (..., n_diphone_classes) containing diphone logits
                       n_diphone_classes = P^2 + 1 where P = mono_n_classes - 1
                       First class (index 0) is BLANK
        mono_n_classes: Number of mono phoneme classes including BLANK (typically 41)
        use_log_space: If True, uses log-sum-exp for numerical stability in log space.
                      If False, converts to probabilities, sums, then converts back to logits.
    
    Returns:
        phoneme_logits: Array of shape (..., mono_n_classes) containing phoneme logits
                       First class (index 0) is BLANK
    """
    is_torch = isinstance(diphone_logits, torch.Tensor)
    
    if is_torch:
        device = diphone_logits.device
        dtype = diphone_logits.dtype
    else:
        device = None
        dtype = diphone_logits.dtype
    
    # P = number of non-blank phoneme classes
    P = mono_n_classes - 1
    
    # Expected number of diphone classes: P^2 + 1 (P^2 diphones + 1 BLANK)
    expected_diphone_classes = P * P + 1
    actual_diphone_classes = diphone_logits.shape[-1]
    
    if actual_diphone_classes != expected_diphone_classes:
        raise ValueError(
            f"Mismatch in diphone classes: expected {expected_diphone_classes} "
            f"(P^2 + 1 where P={P}), but got {actual_diphone_classes}"
        )
    
    # Get original shape
    original_shape = diphone_logits.shape
    batch_shape = original_shape[:-1]
    
    # Reshape to (..., n_diphone_classes)
    diphone_logits_flat = diphone_logits.reshape(-1, actual_diphone_classes)
    n_timesteps = diphone_logits_flat.shape[0]
    
    # Initialize phoneme logits: (n_timesteps, mono_n_classes)
    if is_torch:
        phoneme_logits = torch.zeros(n_timesteps, mono_n_classes, device=device, dtype=dtype)
    else:
        phoneme_logits = np.zeros((n_timesteps, mono_n_classes), dtype=dtype)
    
    # BLANK (index 0) maps directly: diphone BLANK -> phoneme BLANK
    phoneme_logits[:, 0] = diphone_logits_flat[:, 0]
    
    # For each phoneme ph (1-indexed, so ph ranges from 1 to P)
    # Sum all diphone probabilities where next = ph
    for ph in range(1, mono_n_classes):  # ph ranges from 1 to P (phoneme indices)
        # Find all diphone IDs that end with phoneme ph
        # diphone_id = (prev - 1) * P + (ph - 1) + 1
        # For prev from 1 to P:
        diphone_indices = []
        for prev in range(1, P + 1):
            diphone_id = (prev - 1) * P + (ph - 1) + 1
            diphone_indices.append(diphone_id)
        
        # Extract logits for these diphone indices
        diphone_logits_for_ph = diphone_logits_flat[:, diphone_indices]
        
        if use_log_space:
            # Use log-sum-exp for numerical stability
            if is_torch:
                phoneme_logits[:, ph] = torch.logsumexp(diphone_logits_for_ph, dim=-1)
            else:
                phoneme_logits[:, ph] = np.log(np.sum(np.exp(diphone_logits_for_ph), axis=-1))
        else:
            # Convert to probabilities, sum, convert back
            if is_torch:
                probs = torch.softmax(diphone_logits_for_ph, dim=-1)
                summed_probs = torch.sum(probs, dim=-1)
                # Convert back to logits (add small epsilon to avoid log(0))
                eps = 1e-10
                phoneme_logits[:, ph] = torch.log(summed_probs + eps)
            else:
                probs = np.exp(diphone_logits_for_ph - np.max(diphone_logits_for_ph, axis=-1, keepdims=True))
                probs = probs / np.sum(probs, axis=-1, keepdims=True)
                summed_probs = np.sum(probs, axis=-1)
                eps = 1e-10
                phoneme_logits[:, ph] = np.log(summed_probs + eps)
    
    # Reshape back to original batch shape + mono_n_classes
    output_shape = batch_shape + (mono_n_classes,)
    phoneme_logits = phoneme_logits.reshape(output_shape)
    
    return phoneme_logits


def create_diphone_to_phoneme_mapping_matrix(
    mono_n_classes: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Create a mapping matrix M such that phoneme_logits = diphone_logits @ M
    
    This is useful for efficient batch operations.
    
    Args:
        mono_n_classes: Number of mono phoneme classes including BLANK
        device: Device to create the matrix on
    
    Returns:
        mapping_matrix: Tensor of shape (n_diphone_classes, mono_n_classes)
                       where mapping_matrix[i, j] = 1 if diphone i ends with phoneme j, else 0
    """
    P = mono_n_classes - 1
    n_diphone_classes = P * P + 1
    
    # Initialize mapping matrix
    mapping_matrix = torch.zeros(n_diphone_classes, mono_n_classes, device=device)
    
    # BLANK maps to BLANK
    mapping_matrix[0, 0] = 1.0
    
    # For each phoneme ph (1-indexed)
    for ph in range(1, mono_n_classes):
        # Find all diphone IDs that end with phoneme ph
        for prev in range(1, P + 1):
            diphone_id = (prev - 1) * P + (ph - 1) + 1
            mapping_matrix[diphone_id, ph] = 1.0
    
    return mapping_matrix


def marginalize_diphone_logits_vectorized(
    diphone_logits: Union[torch.Tensor, np.ndarray],
    mono_n_classes: int,
    use_log_space: bool = True
) -> Union[torch.Tensor, np.ndarray]:
    """
    Vectorized version using matrix multiplication (faster for large batches).
    
    Args:
        diphone_logits: Array of shape (..., n_diphone_classes) containing diphone logits
        mono_n_classes: Number of mono phoneme classes including BLANK
        use_log_space: If True, uses log-sum-exp. If False, uses probability space.
    
    Returns:
        phoneme_logits: Array of shape (..., mono_n_classes) containing phoneme logits
    """
    is_torch = isinstance(diphone_logits, torch.Tensor)
    
    if is_torch:
        device = diphone_logits.device
        dtype = diphone_logits.dtype
    else:
        device = 'cpu'
        # Convert numpy dtype to torch dtype
        if isinstance(diphone_logits, np.ndarray):
            if diphone_logits.dtype == np.float32:
                dtype = torch.float32
            elif diphone_logits.dtype == np.float64:
                dtype = torch.float64
            else:
                dtype = torch.float32  # Default to float32
        else:
            dtype = torch.float32
        diphone_logits = torch.from_numpy(diphone_logits).to(dtype)
    
    # Create mapping matrix
    mapping_matrix = create_diphone_to_phoneme_mapping_matrix(mono_n_classes, device=device)
    
    # Get original shape
    original_shape = diphone_logits.shape
    batch_shape = original_shape[:-1]
    
    # Reshape to (n_timesteps, n_diphone_classes)
    diphone_logits_flat = diphone_logits.reshape(-1, original_shape[-1])
    n_timesteps = diphone_logits_flat.shape[0]  # Number of timesteps (flattened batch and time)
    
    if use_log_space:
        # Use log-sum-exp for numerical stability
        # For each phoneme, we need to compute max over diphones that map to it, then log-sum-exp
        # We can do this efficiently using the mapping matrix
        # For each timestep and phoneme: log(sum(exp(diphone_logits[diphones_for_phoneme])))
        # We compute this by: max + log(sum(exp(logits - max))) where max is per-phoneme
        # First, compute max per phoneme (over diphones that map to it)
        # This is: max(diphone_logits @ mapping_matrix, but we need to mask zeros)
        # Instead, we'll compute it more directly:
        phoneme_logits_flat = torch.zeros(n_timesteps, mono_n_classes, device=device, dtype=dtype)
        
        # BLANK maps directly
        phoneme_logits_flat[:, 0] = diphone_logits_flat[:, 0]
        
        # For each phoneme (1 to P), compute log-sum-exp over diphones that end with it
        P = mono_n_classes - 1  # Number of non-blank phoneme classes
        for ph in range(1, mono_n_classes):
            # Get diphone indices that map to this phoneme
            diphone_indices = []
            for prev in range(1, P + 1):
                diphone_id = (prev - 1) * P + (ph - 1) + 1
                diphone_indices.append(diphone_id)
            
            # Extract logits for these diphones
            diphone_logits_for_ph = diphone_logits_flat[:, diphone_indices]
            
            # Compute log-sum-exp
            max_logits = torch.max(diphone_logits_for_ph, dim=-1)[0]
            exp_logits = torch.exp(diphone_logits_for_ph - max_logits.unsqueeze(-1))
            summed_exp = torch.sum(exp_logits, dim=-1)
            phoneme_logits_flat[:, ph] = torch.log(summed_exp + 1e-10) + max_logits
    else:
        # Convert to probabilities, then sum
        probs = torch.softmax(diphone_logits_flat, dim=-1)
        phoneme_probs = torch.matmul(probs, mapping_matrix)
        phoneme_logits_flat = torch.log(phoneme_probs + 1e-10)
    
    # Reshape back
    output_shape = batch_shape + (mono_n_classes,)
    phoneme_logits = phoneme_logits_flat.reshape(output_shape)
    
    if not is_torch:
        phoneme_logits = phoneme_logits.numpy()
    
    return phoneme_logits

