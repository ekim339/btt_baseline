"""
SageMaker wrapper for checkpoint metrics evaluation script

This script is called by SageMaker PyTorch estimator.
It reads hyperparameters from SageMaker environment and calls the main evaluation function.
"""

import os
import sys
import json

# SageMaker passes hyperparameters via environment variable
sm_hps = os.environ.get('SM_HPS')
if sm_hps:
    hps = json.loads(sm_hps)
    
    # Extract hyperparameters
    checkpoint_path = hps.get('checkpoint_path')
    data_dir = hps.get('data_dir')
    csv_path = hps.get('csv_path', None)
    gpu_number = int(hps.get('gpu_number', '0'))
    
    # Build command line arguments
    sys.argv = ['evaluate_checkpoint_metrics.py']
    
    sys.argv.extend(['--checkpoint_path', checkpoint_path])
    sys.argv.extend(['--data_dir', data_dir])
    sys.argv.extend(['--gpu_number', str(gpu_number)])
    
    if csv_path:
        sys.argv.extend(['--csv_path', csv_path])
    
    print(f"[evaluate_metrics_sagemaker] Parsed hyperparameters:")
    print(f"  checkpoint_path: {checkpoint_path}")
    print(f"  data_dir: {data_dir}")
    print(f"  csv_path: {csv_path}")
    print(f"  gpu_number: {gpu_number}")
    print()
else:
    print("[evaluate_metrics_sagemaker] ERROR: SM_HPS environment variable not found")
    sys.exit(1)

# Import and run the main evaluation script
if __name__ == '__main__':
    # Import the main evaluation module
    # The argparse will parse sys.argv which we've already set up
    import evaluate_checkpoint_metrics
    # Call main function (it will use the args we set in sys.argv)
    evaluate_checkpoint_metrics.main()

