"""
SageMaker wrapper for ensemble inference script

This script is called by SageMaker PyTorch estimator.
It reads hyperparameters from SageMaker environment and calls the main inference function.
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
    data_dir = hps.get('data_dir')
    csv_path = hps.get('csv_path', None)
    eval_type = hps.get('eval_type', 'test')
    output_dir = hps.get('output_dir', None)
    ensemble_method = hps.get('ensemble_method', 'average_logits')
    gpu_number = int(hps.get('gpu_number', '0'))
    use_language_model = hps.get('use_language_model', 'false').lower() == 'true'
    
    # Build command line arguments
    sys.argv = ['ensemble_inference.py']
    
    # Add checkpoint paths (nargs='+' expects all values after a single flag)
    sys.argv.extend(['--checkpoint_paths'] + checkpoint_paths)
    
    # Add other arguments
    sys.argv.extend(['--data_dir', data_dir])
    sys.argv.extend(['--eval_type', eval_type])
    sys.argv.extend(['--ensemble_method', ensemble_method])
    sys.argv.extend(['--gpu_number', str(gpu_number)])
    
    if csv_path:
        sys.argv.extend(['--csv_path', csv_path])
    
    if output_dir:
        sys.argv.extend(['--output_dir', output_dir])
    
    if use_language_model:
        sys.argv.append('--use_language_model')
    
    print(f"[ensemble_inference_sagemaker] Parsed hyperparameters:")
    print(f"  checkpoint_paths: {checkpoint_paths}")
    print(f"  data_dir: {data_dir}")
    print(f"  eval_type: {eval_type}")
    print(f"  ensemble_method: {ensemble_method}")
    print(f"  gpu_number: {gpu_number}")
    print(f"  output_dir: {output_dir}")
    print(f"  use_language_model: {use_language_model}")
    print()
else:
    print("[ensemble_inference_sagemaker] ERROR: SM_HPS environment variable not found")
    sys.exit(1)

# Import and run the main inference script
if __name__ == '__main__':
    # Import the main inference module
    # The argparse will parse sys.argv which we've already set up
    import ensemble_inference
    # Call main function (it will use the args we set in sys.argv)
    ensemble_inference.main()

