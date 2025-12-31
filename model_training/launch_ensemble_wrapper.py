"""
Launch SageMaker PyTorch estimator to create ensemble model checkpoint
"""

import sagemaker
from sagemaker.pytorch import PyTorch
import json
from typing import List, Optional


def launch_ensemble_wrapper(
    checkpoint_paths: List[str],
    output_path: str,
    instance_type: str = 'ml.g5.8xlarge',  # GPU instance required
    gpu_number: int = 0,
    val_PER: float = 0.0,
    val_loss: float = 0.0,
    source_dir: str = './model_training',
    role: Optional[str] = None,
) -> PyTorch:
    """
    Launch SageMaker PyTorch estimator to create ensemble model checkpoint.
    
    Args:
        checkpoint_paths: List of S3 paths to checkpoint directories or files
        output_path: S3 path where ensemble checkpoint will be saved
        instance_type: SageMaker instance type (must have GPU)
        gpu_number: GPU number to use
        val_PER: Validation PER (for metadata)
        val_loss: Validation loss (for metadata)
        source_dir: Local directory containing the wrapper script
        role: SageMaker execution role (defaults to get_execution_role())
    
    Returns:
        PyTorch estimator instance
    """
    if role is None:
        role = sagemaker.get_execution_role()
    
    # Build hyperparameters
    hyperparameters = {
        'checkpoint_paths': json.dumps(checkpoint_paths),  # JSON encode list
        'output_path': output_path,
        'gpu_number': str(gpu_number),
        'val_PER': str(val_PER),
        'val_loss': str(val_loss),
    }
    
    # Create PyTorch estimator
    estimator = PyTorch(
        entry_point='ensemble_wrapper_sagemaker.py',  # Wrapper script for SageMaker
        source_dir=source_dir,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        framework_version='1.9.0',
        py_version='py38',
        
        # Set max run time (should be fast - just loading models and saving)
        max_run=3600,  # 1 hour should be enough
        
        # Dependencies
        dependencies=[f'{source_dir}/requirements.txt'],
        
        # Hyperparameters
        hyperparameters=hyperparameters,
        
        # Output path (for logs)
        output_path=output_path.rsplit('/', 1)[0] if '/' in output_path else output_path,
    )
    
    return estimator


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch SageMaker job to create ensemble checkpoint')
    parser.add_argument('--checkpoint_paths', nargs='+', required=True,
                       help='S3 paths to individual model checkpoints')
    parser.add_argument('--output_path', type=str, required=True,
                       help='S3 path to save ensemble checkpoint')
    parser.add_argument('--instance_type', type=str, default='ml.g5.8xlarge',
                       help='SageMaker instance type')
    parser.add_argument('--gpu_number', type=int, default=0,
                       help='GPU number to use')
    parser.add_argument('--val_PER', type=float, default=0.0,
                       help='Validation PER (for metadata)')
    parser.add_argument('--val_loss', type=float, default=0.0,
                       help='Validation loss (for metadata)')
    parser.add_argument('--source_dir', type=str, default='./model_training',
                       help='Local directory containing the wrapper script')
    
    args = parser.parse_args()
    
    estimator = launch_ensemble_wrapper(
        checkpoint_paths=args.checkpoint_paths,
        output_path=args.output_path,
        instance_type=args.instance_type,
        gpu_number=args.gpu_number,
        val_PER=args.val_PER,
        val_loss=args.val_loss,
        source_dir=args.source_dir,
    )
    
    print(f"Launching SageMaker job to create ensemble checkpoint...")
    print(f"Checkpoint paths: {args.checkpoint_paths}")
    print(f"Output path: {args.output_path}")
    print(f"Instance type: {args.instance_type}")
    
    estimator.fit(wait=True)
    
    print("Ensemble checkpoint creation completed!")

