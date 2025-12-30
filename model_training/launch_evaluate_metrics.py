"""
Launch SageMaker PyTorch Estimator for Checkpoint Metrics Evaluation

This script launches a SageMaker training job that evaluates checkpoint metrics (DER and PER).
"""

import sagemaker
from sagemaker.pytorch import PyTorch
import json

def launch_evaluate_metrics(
    checkpoint_path: str,
    data_dir: str,
    csv_path: str = None,
    gpu_number: int = 0,
    instance_type: str = 'ml.g5.xlarge',  # GPU instance
    source_dir: str = '/home/ec2-user/SageMaker/btt_baseline/model_training',
    role: str = None,
):
    """
    Launch SageMaker PyTorch estimator for checkpoint metrics evaluation.
    
    Args:
        checkpoint_path: S3 or local path to checkpoint directory or file
        data_dir: S3 path to dataset directory
        csv_path: Optional S3 path to CSV metadata file
        gpu_number: GPU number to use
        instance_type: SageMaker instance type (must have GPU)
        source_dir: Local directory containing the evaluation script
        role: SageMaker execution role (defaults to get_execution_role())
    """
    if role is None:
        role = sagemaker.get_execution_role()
    
    # Build hyperparameters
    hyperparameters = {
        'checkpoint_path': checkpoint_path,
        'data_dir': data_dir,
        'gpu_number': str(gpu_number),
    }
    
    if csv_path:
        hyperparameters['csv_path'] = csv_path
    
    # Create PyTorch estimator
    estimator = PyTorch(
        entry_point='evaluate_metrics_sagemaker.py',  # Wrapper script for SageMaker
        source_dir=source_dir,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        framework_version='1.9.0',
        py_version='py38',
        
        # Set max run time (evaluation should be faster than training)
        max_run=3600,  # 1 hour should be enough
        
        # Dependencies
        dependencies=[f'{source_dir}/requirements.txt'],
        
        # Hyperparameters
        hyperparameters=hyperparameters,
        
        # Output path (for logs)
        output_path=checkpoint_path if checkpoint_path.startswith('s3://') else None,
    )
    
    return estimator


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch SageMaker checkpoint metrics evaluation job')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='S3 path to checkpoint directory or file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='S3 path to dataset directory')
    parser.add_argument('--csv_path', type=str, default=None,
                       help='Optional S3 path to CSV metadata file')
    parser.add_argument('--instance_type', type=str, default='ml.g5.xlarge',
                       help='SageMaker instance type (must have GPU)')
    parser.add_argument('--gpu_number', type=int, default=0,
                       help='GPU number to use')
    parser.add_argument('--source_dir', type=str,
                       default='/home/ec2-user/SageMaker/btt_baseline/model_training',
                       help='Local directory containing evaluation script')
    parser.add_argument('--wait', action='store_true',
                       help='Wait for job to complete')
    
    args = parser.parse_args()
    
    # Launch estimator
    estimator = launch_evaluate_metrics(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        instance_type=args.instance_type,
        gpu_number=args.gpu_number,
        source_dir=args.source_dir,
    )
    
    # Fit (no input data needed, but SageMaker requires fit())
    print(f"Launching SageMaker metrics evaluation job...")
    print(f"Checkpoint path: {args.checkpoint_path}")
    print(f"Data dir: {args.data_dir}")
    print(f"Instance type: {args.instance_type}")
    
    estimator.fit(wait=args.wait)
    
    print(f"Job name: {estimator.latest_training_job.name}")
    print(f"Job status: {estimator.latest_training_job.describe()['TrainingJobStatus']}")

