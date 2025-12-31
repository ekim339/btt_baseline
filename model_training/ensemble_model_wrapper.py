"""
Ensemble Model Wrapper

This module creates a wrapper model that loads multiple checkpoints and averages their outputs.
This wrapper can be saved as a checkpoint for later use.
"""

import torch
import torch.nn as nn
from typing import List
from rnn_model import GRUDecoder
from omegaconf import OmegaConf
import s3fs
import os


class EnsembleModelWrapper(nn.Module):
    """
    Wrapper model that loads multiple models and averages their outputs.
    
    This allows you to save an ensemble as a single checkpoint.
    Note: All models must have the same architecture.
    """
    
    def __init__(self, checkpoint_paths: List[str], device: torch.device):
        """
        Initialize ensemble wrapper with multiple checkpoints.
        
        Args:
            checkpoint_paths: List of paths to checkpoint files/directories
            device: Device to load models on
        """
        super(EnsembleModelWrapper, self).__init__()
        
        self.models = nn.ModuleList()
        self.device = device
        self.n_models = len(checkpoint_paths)
        
        # Load each model
        for i, checkpoint_path in enumerate(checkpoint_paths):
            print(f"Loading model {i+1}/{self.n_models} from {checkpoint_path}...")
            model = self._load_model_from_checkpoint(checkpoint_path, device)
            model.eval()  # Set to evaluation mode
            self.models.append(model)
        
        print(f"Loaded {self.n_models} models for ensembling")
    
    def _load_model_from_checkpoint(self, checkpoint_path: str, device: torch.device) -> GRUDecoder:
        """Load a single model from checkpoint."""
        fs = s3fs.S3FileSystem() if checkpoint_path.startswith('s3://') else None
        
        # Determine checkpoint file location
        if checkpoint_path.endswith('best_checkpoint') or checkpoint_path.endswith('final_checkpoint'):
            checkpoint_file = checkpoint_path
            checkpoint_dir = os.path.dirname(checkpoint_path.rstrip('/'))
        else:
            checkpoint_dir = checkpoint_path.rstrip('/')
            checkpoint_file = os.path.join(checkpoint_dir, 'best_checkpoint')
            if checkpoint_path.startswith('s3://'):
                if not fs.exists(checkpoint_file):
                    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint/best_checkpoint')
            else:
                if not os.path.exists(checkpoint_file):
                    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint/best_checkpoint')
        
        # Load args.yaml
        args_path_candidates = [
            os.path.join(checkpoint_dir, 'args.yaml'),
            os.path.join(checkpoint_dir, 'checkpoint/args.yaml'),
        ]
        
        model_args = None
        for args_path in args_path_candidates:
            try:
                if checkpoint_path.startswith('s3://'):
                    with fs.open(args_path, 'rb') as f:
                        model_args = OmegaConf.load(f)
                else:
                    if os.path.exists(args_path):
                        model_args = OmegaConf.load(args_path)
                if model_args:
                    break
            except:
                continue
        
        if model_args is None:
            raise FileNotFoundError(f"Could not find args.yaml for checkpoint: {checkpoint_path}")
        
        # Create model
        model = GRUDecoder(
            neural_dim=model_args['model']['n_input_features'],
            n_units=model_args['model']['n_units'],
            n_days=len(model_args['dataset']['sessions']),
            n_classes=model_args['dataset']['n_classes'],
            rnn_dropout=model_args['model']['rnn_dropout'],
            input_dropout=model_args['model']['input_network']['input_layer_dropout'],
            n_layers=model_args['model']['n_layers'],
            patch_size=model_args['model']['patch_size'],
            patch_stride=model_args['model']['patch_stride'],
        )
        
        # Load checkpoint
        if checkpoint_path.startswith('s3://'):
            with fs.open(checkpoint_file, 'rb') as f:
                checkpoint = torch.load(f, weights_only=False, map_location='cpu')
        else:
            checkpoint = torch.load(checkpoint_file, weights_only=False, map_location='cpu')
        
        # Handle compatibility: convert old ParameterList format to single tensor format
        model_state_dict = checkpoint['model_state_dict']
        if 'day_weights' not in model_state_dict:
            # Old format: ParameterList with day_weights.0, day_weights.1, etc.
            day_weights_list = []
            day_biases_list = []
            i = 0
            while f'day_weights.{i}' in model_state_dict:
                day_weights_list.append(model_state_dict[f'day_weights.{i}'])
                if f'day_biases.{i}' in model_state_dict:
                    day_biases_list.append(model_state_dict[f'day_biases.{i}'])
                i += 1
            
            if day_weights_list:
                model_state_dict['day_weights'] = torch.stack(day_weights_list, dim=0)
                if day_biases_list:
                    model_state_dict['day_biases'] = torch.stack(day_biases_list, dim=0)
                
                for j in range(i):
                    if f'day_weights.{j}' in model_state_dict:
                        del model_state_dict[f'day_weights.{j}']
                    if f'day_biases.{j}' in model_state_dict:
                        del model_state_dict[f'day_biases.{j}']
        
        # Remove "module." prefix if present
        for key in list(model_state_dict.keys()):
            if key.startswith("module."):
                model_state_dict[key.replace("module.", "")] = model_state_dict.pop(key)
            if key.startswith("_orig_mod."):
                model_state_dict[key.replace("_orig_mod.", "")] = model_state_dict.pop(key)
        
        model.load_state_dict(model_state_dict)
        model.to(device)
        
        return model
    
    def forward(self, x, day_idx, lengths=None, states=None, return_state=False):
        """
        Forward pass: average logits from all models.
        
        Args:
            x: Input features [B, T, D]
            day_idx: Day indices [B]
            lengths: Optional sequence lengths
            states: Optional initial states
            return_state: Whether to return hidden states
        
        Returns:
            Averaged logits [B, T, C] or (logits, states) if return_state=True
        """
        # Get logits from all models
        all_logits = []
        all_states = []
        
        for model in self.models:
            if return_state:
                logits, states = model(x, day_idx, lengths, states, return_state=True)
                all_logits.append(logits)
                all_states.append(states)
            else:
                logits = model(x, day_idx, lengths, states, return_state=False)
                all_logits.append(logits)
        
        # Average logits
        ensemble_logits = torch.stack(all_logits, dim=0).mean(dim=0)  # [B, T, C]
        
        if return_state:
            # Average states (if needed)
            # Note: This is a simple average - you might want more sophisticated averaging
            ensemble_states = None
            if all_states and all_states[0] is not None:
                ensemble_states = torch.stack(all_states, dim=0).mean(dim=0)
            return ensemble_logits, ensemble_states
        
        return ensemble_logits


def save_ensemble_checkpoint(
    checkpoint_paths: List[str],
    output_path: str,
    device: torch.device,
    val_PER: float = 0.0,
    val_loss: float = 0.0
):
    """
    Create and save an ensemble model as a checkpoint.
    
    Args:
        checkpoint_paths: List of paths to individual model checkpoints
        output_path: Path to save the ensemble checkpoint
        device: Device to load models on
        val_PER: Validation PER (for metadata)
        val_loss: Validation loss (for metadata)
    """
    # Create ensemble wrapper
    ensemble_model = EnsembleModelWrapper(checkpoint_paths, device)
    
    # Create checkpoint dictionary
    checkpoint = {
        'model_state_dict': ensemble_model.state_dict(),
        'val_PER': val_PER,
        'val_loss': val_loss,
        'n_models': len(checkpoint_paths),
        'checkpoint_paths': checkpoint_paths,
        'ensemble_type': 'wrapper'
    }
    
    # Save checkpoint
    fs = s3fs.S3FileSystem() if output_path.startswith('s3://') else None
    
    if output_path.startswith('s3://'):
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            torch.save(checkpoint, tmp_file.name)
            tmp_file.flush()
            with open(tmp_file.name, 'rb') as f:
                with fs.open(output_path, 'wb') as s3_file:
                    s3_file.write(f.read())
            os.unlink(tmp_file.name)
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(checkpoint, output_path)
    
    print(f"Saved ensemble checkpoint to: {output_path}")
    return ensemble_model


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create ensemble model checkpoint')
    parser.add_argument('--checkpoint_paths', nargs='+', required=True,
                       help='Paths to individual model checkpoints')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save ensemble checkpoint')
    parser.add_argument('--gpu_number', type=int, default=0,
                       help='GPU number to use')
    
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu_number}' if torch.cuda.is_available() else 'cpu')
    
    save_ensemble_checkpoint(
        checkpoint_paths=args.checkpoint_paths,
        output_path=args.output_path,
        device=device
    )

