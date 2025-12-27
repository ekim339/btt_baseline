"""
SageMaker Training Launcher

Complete script to launch SageMaker PyTorch training job.
"""

import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role

# Get SageMaker execution role
# If running in SageMaker notebook, this will automatically get the role
# Otherwise, specify the role ARN manually:
# role = 'arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole'
role = get_execution_role()

# Create PyTorch estimator
estimator = PyTorch(
    entry_point='train_model.py',              # Your training script
    source_dir='model_training',               # Directory with training code
    role=role,
    instance_count=1,
    instance_type='ml.g4dn.16xlarge',          # GPU instance
    framework_version='2.0.1',                 # PyTorch version
    py_version='py310',                        # Python version
    max_run=259200,                            # Max time: 72 hours (in seconds)
    base_job_name='btt-training',
    hyperparameters={
        "config": "rnn_args.yaml"              # or "rnn_args_diphone.yaml"
    },
    output_path='s3://your-bucket/model-woody/'
)

# Start training
# The 'training' key maps to SM_CHANNEL_TRAINING environment variable
estimator.fit({
    'training': 's3://your-bucket/data/hdf5_data_final'
})

# To run without waiting:
# estimator.fit({'training': 's3://your-bucket/data/hdf5_data_final'}, wait=False)

print(f"Training completed!")
print(f"Model artifacts: {estimator.model_data}")
print(f"Job name: {estimator.latest_training_job.name}")

