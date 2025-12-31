"""
SageMaker wrapper for ensemble model wrapper script

This script is called by SageMaker PyTorch estimator.
It reads hyperparameters from SageMaker environment and creates an ensemble checkpoint.
"""

import os
import sys
import json

# SageMaker passes hyperparameters via environment variable
sm_hps = os.environ.get('SM_HPS')
if sm_hps:
    hps = json.loads(sm_hps)
    
    # Extract hyperparameters
    checkpoint_paths = json.loads(hps.get('checkpoint_paths', '[]'))
    output_path = hps.get('output_path')
    gpu_number = int(hps.get('gpu_number', '0'))
    val_PER = float(hps.get('val_PER', '0.0'))
    val_loss = float(hps.get('val_loss', '0.0'))
    
    print(f"[ensemble_wrapper_sagemaker] Parsed hyperparameters:")
    print(f"  checkpoint_paths: {checkpoint_paths}")
    print(f"  output_path: {output_path}")
    print(f"  gpu_number: {gpu_number}")
    print(f"  val_PER: {val_PER}")
    print(f"  val_loss: {val_loss}")
    print()
    
    # Import and run the ensemble wrapper
    from ensemble_model_wrapper import save_ensemble_checkpoint
    import torch
    
    device = torch.device(f'cuda:{gpu_number}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create and save ensemble checkpoint
    save_ensemble_checkpoint(
        checkpoint_paths=checkpoint_paths,
        output_path=output_path,
        device=device,
        val_PER=val_PER,
        val_loss=val_loss
    )
    
    print("Ensemble checkpoint creation completed successfully!")
else:
    print("[ensemble_wrapper_sagemaker] ERROR: SM_HPS environment variable not found")
    sys.exit(1)

