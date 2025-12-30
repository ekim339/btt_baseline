"""
Launch SageMaker PyTorch Estimator for Ensemble Inference

This script launches a SageMaker training job that runs ensemble inference.
Since SageMaker doesn't have a dedicated inference estimator, we use the PyTorch estimator
in "training" mode to run inference.
"""

import sagemaker
from sagemaker.pytorch import PyTorch
import json

def launch_ensemble_inference(
    checkpoint_paths: list,
    data_dir: str,
    csv_path: str = None,
    eval_type: str = 'test',
    output_dir: str = None,
    ensemble_method: str = 'average_logits',
    use_language_model: bool = False,
    instance_type: str = 'ml.g5.xlarge',  # GPU instance
    gpu_number: int = 0,
    source_dir: str = '/home/ec2-user/SageMaker/btt_baseline/model_training',
    role: str = None,
):
    """
    Launch SageMaker PyTorch estimator for ensemble inference.
    
    Args:
        checkpoint_paths: List of S3 paths to checkpoint directories or files
        data_dir: S3 path to dataset directory
        csv_path: Optional S3 path to CSV metadata file
        eval_type: 'test' or 'val'
        output_dir: S3 path for output (defaults to first checkpoint directory)
        ensemble_method: 'average_logits' or 'average_probs'
        use_language_model: Whether to use language model
        instance_type: SageMaker instance type (must have GPU)
        gpu_number: GPU number to use
        source_dir: Local directory containing the inference script
        role: SageMaker execution role (defaults to get_execution_role())
    """
    if role is None:
        role = sagemaker.get_execution_role()
    
    # Build hyperparameters
    hyperparameters = {
        'checkpoint_paths': json.dumps(checkpoint_paths),  # JSON encode list
        'data_dir': data_dir,
        'eval_type': eval_type,
        'ensemble_method': ensemble_method,
        'gpu_number': str(gpu_number),
    }
    
    if csv_path:
        hyperparameters['csv_path'] = csv_path
    
    if output_dir:
        hyperparameters['output_dir'] = output_dir
    
    if use_language_model:
        hyperparameters['use_language_model'] = 'true'
    
    # Create PyTorch estimator
    estimator = PyTorch(
        entry_point='ensemble_inference_sagemaker.py',  # Wrapper script for SageMaker
        source_dir=source_dir,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        framework_version='1.9.0',
        py_version='py38',
        
        # Set max run time (inference should be faster than training)
        max_run=3600,  # 1 hour should be enough for inference
        
        # Dependencies
        dependencies=[f'{source_dir}/requirements.txt'],
        
        # Hyperparameters
        hyperparameters=hyperparameters,
        
        # Output path (for logs and any outputs)
        output_path=output_dir if output_dir else checkpoint_paths[0],
    )
    
    return estimator


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch SageMaker ensemble inference job')
    parser.add_argument('--checkpoint_paths', nargs='+', required=True,
                       help='S3 paths to checkpoint directories or files')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='S3 path to dataset directory')
    parser.add_argument('--csv_path', type=str, default=None,
                       help='Optional S3 path to CSV metadata file')
    parser.add_argument('--eval_type', type=str, default='test', choices=['test', 'val'],
                       help='Evaluation type: test or val')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='S3 path for output (defaults to first checkpoint directory)')
    parser.add_argument('--ensemble_method', type=str, default='average_logits',
                       choices=['average_logits', 'average_probs'],
                       help='Ensemble method')
    parser.add_argument('--use_language_model', action='store_true',
                       help='Use language model for final predictions')
    parser.add_argument('--instance_type', type=str, default='ml.g5.xlarge',
                       help='SageMaker instance type (must have GPU)')
    parser.add_argument('--gpu_number', type=int, default=0,
                       help='GPU number to use')
    parser.add_argument('--source_dir', type=str,
                       default='/home/ec2-user/SageMaker/btt_baseline/model_training',
                       help='Local directory containing inference script')
    parser.add_argument('--wait', action='store_true',
                       help='Wait for job to complete')
    
    args = parser.parse_args()
    
    # Launch estimator
    estimator = launch_ensemble_inference(
        checkpoint_paths=args.checkpoint_paths,
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        eval_type=args.eval_type,
        output_dir=args.output_dir,
        ensemble_method=args.ensemble_method,
        use_language_model=args.use_language_model,
        instance_type=args.instance_type,
        gpu_number=args.gpu_number,
        source_dir=args.source_dir,
    )
    
    # Fit (no input data needed for inference, but SageMaker requires fit())
    print(f"Launching SageMaker inference job...")
    print(f"Checkpoint paths: {args.checkpoint_paths}")
    print(f"Data dir: {args.data_dir}")
    print(f"Instance type: {args.instance_type}")
    print(f"Output dir: {args.output_dir or args.checkpoint_paths[0]}")
    
    estimator.fit(wait=args.wait)
    
    print(f"Job name: {estimator.latest_training_job.name}")
    print(f"Job status: {estimator.latest_training_job.describe()['TrainingJobStatus']}")

