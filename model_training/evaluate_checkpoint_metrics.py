"""
Evaluate Checkpoint Metrics Script

This script loads a checkpoint and evaluates it on validation data,
printing DER (Diphone Error Rate) and PER (Phoneme Error Rate) to the log.

For PER calculation, diphone log probabilities are marginalized to phoneme probabilities
by summing all diphones that end with the same phoneme.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import argparse
import s3fs
import h5py
from omegaconf import OmegaConf
from tqdm import tqdm

from rnn_model import GRUDecoder
from evaluate_model_helpers import runSingleDecodingStep, LOGIT_TO_PHONEME, load_h5py_file
from diphone_to_phoneme_marginalization import marginalize_diphone_logits_vectorized

fs = s3fs.S3FileSystem(anon=False)

# --------------------------------------------------------------------------------
# Argument parser
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Evaluate checkpoint metrics (DER and PER) on validation data.'
)
parser.add_argument(
    '--checkpoint_path',
    type=str,
    required=True,
    help='S3 or local path to checkpoint directory or checkpoint file'
)
parser.add_argument(
    '--data_dir',
    type=str,
    required=True,
    help='S3 or local path to dataset directory'
)
parser.add_argument(
    '--csv_path',
    type=str,
    default=None,
    help='Optional path to CSV metadata file (only needed for corpus info)'
)
parser.add_argument(
    '--gpu_number',
    type=int,
    default=0,
    help='GPU number to use for inference. -1 for CPU.'
)
args = parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Load a model from a checkpoint directory or checkpoint file.
    
    Returns:
        model: Loaded GRUDecoder model
        model_args: Model configuration (OmegaConf)
        is_diphone: Whether this is a diphone model
        mono_n_classes: Number of mono phoneme classes (if diphone model)
    """
    # Determine if checkpoint_path points to a file or directory
    checkpoint_file_names = ['best_checkpoint', 'final_checkpoint', 'checkpoint']
    is_checkpoint_file = any(checkpoint_path.endswith(name) for name in checkpoint_file_names)
    
    if is_checkpoint_file:
        checkpoint_dir = os.path.dirname(checkpoint_path.rstrip('/'))
        checkpoint_file = checkpoint_path
        args_path_candidates = [
            os.path.join(checkpoint_dir, 'args.yaml'),
            os.path.join(os.path.dirname(checkpoint_dir), 'checkpoint/args.yaml'),
            os.path.join(checkpoint_dir, 'checkpoint/args.yaml'),
        ]
    else:
        checkpoint_dir = checkpoint_path.rstrip('/')
        checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint/best_checkpoint')
        args_path_candidates = [
            os.path.join(checkpoint_dir, 'checkpoint/args.yaml'),
            os.path.join(checkpoint_dir, 'args.yaml'),
        ]
    
    # Load args.yaml
    model_args = None
    args_path = None
    for candidate_path in args_path_candidates:
        try:
            if checkpoint_dir.startswith('s3://') or candidate_path.startswith('s3://'):
                if fs.exists(candidate_path):
                    with fs.open(candidate_path, 'rb') as f:
                        model_args = OmegaConf.load(f)
                    args_path = candidate_path
                    break
            else:
                if os.path.exists(candidate_path):
                    model_args = OmegaConf.load(candidate_path)
                    args_path = candidate_path
                    break
        except Exception as e:
            continue
    
    if model_args is None:
        raise FileNotFoundError(f"Could not find args.yaml. Tried: {args_path_candidates}")
    
    print(f"Loaded model args from: {args_path}")
    
    # Determine if this is a diphone model
    n_classes = model_args['dataset']['n_classes']
    mono_n_classes = model_args['dataset'].get('mono_n_classes', None)
    
    is_diphone = False
    if mono_n_classes is not None:
        P = mono_n_classes - 1
        expected_diphone_classes = P * P + 1
        if n_classes == expected_diphone_classes:
            is_diphone = True
    elif n_classes > 100:
        P = int(np.sqrt(n_classes - 1))
        if P * P + 1 == n_classes:
            mono_n_classes = P + 1
            is_diphone = True
    
    # Create model
    model = GRUDecoder(
        neural_dim=model_args['model']['n_input_features'],
        n_units=model_args['model']['n_units'],
        n_days=len(model_args['dataset']['sessions']),
        n_classes=n_classes,
        rnn_dropout=model_args['model']['rnn_dropout'],
        input_dropout=model_args['model']['input_network']['input_layer_dropout'],
        n_layers=model_args['model']['n_layers'],
        patch_size=model_args['model']['patch_size'],
        patch_stride=model_args['model']['patch_stride'],
    )
    
    # Load checkpoint
    if checkpoint_file.startswith('s3://'):
        if not fs.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        with fs.open(checkpoint_file, 'rb') as f:
            checkpoint = torch.load(f, map_location=device, weights_only=False)
    else:
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    
    # Rename keys to remove "module." prefix
    for key in list(checkpoint['model_state_dict'].keys()):
        if key.startswith("module."):
            checkpoint['model_state_dict'][key.replace("module.", "")] = checkpoint['model_state_dict'].pop(key)
        if key.startswith("_orig_mod."):
            checkpoint['model_state_dict'][key.replace("_orig_mod.", "")] = checkpoint['model_state_dict'].pop(key)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from: {checkpoint_file}")
    print(f"Model type: {'diphone' if is_diphone else 'mono'}")
    if is_diphone:
        print(f"Mono classes: {mono_n_classes}")
    print(f"Total classes: {n_classes}")
    
    return model, model_args, is_diphone, mono_n_classes


def calculate_edit_distance(pred_seq, true_seq):
    """Calculate edit distance between two sequences."""
    # Simple edit distance calculation
    m, n = len(pred_seq), len(true_seq)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_seq[i-1] == true_seq[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]


def main():
    checkpoint_path = args.checkpoint_path
    data_dir = args.data_dir
    
    # Set up GPU device
    gpu_number = args.gpu_number
    if torch.cuda.is_available() and gpu_number >= 0:
        if gpu_number >= torch.cuda.device_count():
            raise ValueError(f'GPU number {gpu_number} is out of range. Available GPUs: {torch.cuda.device_count()}')
        device = torch.device(f'cuda:{gpu_number}')
        print(f'Using {device} for model inference.')
    else:
        if gpu_number >= 0:
            print(f'GPU number {gpu_number} requested but not available.')
        print('Using CPU for model inference.')
        device = torch.device('cpu')
    
    # Load model
    print("=" * 80)
    print("LOADING CHECKPOINT")
    print("=" * 80)
    model, model_args, is_diphone, mono_n_classes = load_model_from_checkpoint(checkpoint_path, device)
    print()
    
    # Load CSV file (optional)
    b2txt_csv_df = None
    if args.csv_path:
        if args.csv_path.startswith('s3://'):
            with fs.open(args.csv_path, 'rb') as f:
                b2txt_csv_df = pd.read_csv(f)
        else:
            b2txt_csv_df = pd.read_csv(args.csv_path)
        print(f"Loaded CSV metadata file: {args.csv_path}")
    
    # Load validation data
    print("=" * 80)
    print("LOADING VALIDATION DATA")
    print("=" * 80)
    val_data = {}
    total_val_trials = 0
    
    for session in model_args['dataset']['sessions']:
        val_file = os.path.join(data_dir, session, 'data_val.hdf5')
        
        if data_dir.startswith('s3://'):
            if fs.exists(val_file):
                if b2txt_csv_df is not None:
                    data = load_h5py_file(val_file, b2txt_csv_df)
                else:
                    # Load without CSV (no corpus info)
                    with fs.open(val_file, 'rb') as f:
                        with h5py.File(f, 'r') as h5f:
                            data = {
                                'neural_features': [],
                                'seq_class_ids': [],
                                'mono_seq_class_ids': [],  # Original phoneme labels (for PER)
                                'seq_len': [],
                                'mono_seq_len': [],  # Original phoneme sequence length
                                'session': [],
                                'block_num': [],
                                'trial_num': [],
                            }
                            keys = list(h5f.keys())
                            for key in keys:
                                g = h5f[key]
                                data['neural_features'].append(g['input_features'][:])
                                data['seq_class_ids'].append(g['seq_class_ids'][:] if 'seq_class_ids' in g else None)
                                data['mono_seq_class_ids'].append(g['mono_seq_class_ids'][:] if 'mono_seq_class_ids' in g else None)
                                data['seq_len'].append(g.attrs['seq_len'] if 'seq_len' in g.attrs else None)
                                data['mono_seq_len'].append(g.attrs['mono_seq_len'] if 'mono_seq_len' in g.attrs else None)
                                data['session'].append(g.attrs['session'])
                                data['block_num'].append(g.attrs['block_num'])
                                data['trial_num'].append(g.attrs['trial_num'])
                
                val_data[session] = data
                total_val_trials += len(data['neural_features'])
                print(f'Loaded {len(data["neural_features"])} validation trials for session {session}.')
        else:
            if os.path.exists(val_file):
                if b2txt_csv_df is not None:
                    data = load_h5py_file(val_file, b2txt_csv_df)
                else:
                    with h5py.File(val_file, 'r') as h5f:
                        data = {
                            'neural_features': [],
                            'seq_class_ids': [],
                            'mono_seq_class_ids': [],  # Original phoneme labels (for PER)
                            'seq_len': [],
                            'mono_seq_len': [],  # Original phoneme sequence length
                            'session': [],
                            'block_num': [],
                            'trial_num': [],
                        }
                        keys = list(h5f.keys())
                        for key in keys:
                            g = h5f[key]
                            data['neural_features'].append(g['input_features'][:])
                            data['seq_class_ids'].append(g['seq_class_ids'][:] if 'seq_class_ids' in g else None)
                            data['mono_seq_class_ids'].append(g['mono_seq_class_ids'][:] if 'mono_seq_class_ids' in g else None)
                            data['seq_len'].append(g.attrs['seq_len'] if 'seq_len' in g.attrs else None)
                            data['mono_seq_len'].append(g.attrs['mono_seq_len'] if 'mono_seq_len' in g.attrs else None)
                            data['session'].append(g.attrs['session'])
                            data['block_num'].append(g.attrs['block_num'])
                            data['trial_num'].append(g.attrs['trial_num'])
                
                val_data[session] = data
                total_val_trials += len(data['neural_features'])
                print(f'Loaded {len(data["neural_features"])} validation trials for session {session}.')
    
    print(f"Total number of validation trials: {total_val_trials}")
    print()
    
    # Run inference
    print("=" * 80)
    print("RUNNING INFERENCE")
    print("=" * 80)
    
    all_diphone_preds = []
    all_phoneme_preds = []
    all_diphone_true = []
    all_phoneme_true = []
    
    with tqdm(total=total_val_trials, desc='Running inference', unit='trial') as pbar:
        for session, data in val_data.items():
            input_layer = model_args['dataset']['sessions'].index(session)
            
            for trial in range(len(data['neural_features'])):
                neural_input = data['neural_features'][trial]
                neural_input = np.expand_dims(neural_input, axis=0)
                neural_input = torch.tensor(neural_input, device=device, dtype=torch.bfloat16)
                
                # Get logits
                with torch.no_grad():
                    logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)
                    logits = logits[0].cpu().numpy()  # Remove batch dimension, shape: (time, n_classes)
                
                # Get true labels
                # For diphone models, HDF5 files store original phoneme sequence in seq_class_ids
                # For mono models, seq_class_ids contains phoneme sequence
                true_seq = data['seq_class_ids'][trial]
                seq_len = data['seq_len'][trial]
                if true_seq is not None and seq_len is not None:
                    true_seq = true_seq[:seq_len]
                else:
                    pbar.update(1)
                    continue
                
                if is_diphone:
                    # Diphone model
                    # HDF5 files store original phoneme sequence in seq_class_ids
                    # We need to convert it to diphone sequence for DER calculation
                    phoneme_seq = true_seq
                    
                    # Convert phoneme sequence to diphone sequence for DER
                    # Remove blanks and convert to diphones
                    phoneme_no_blank = phoneme_seq[phoneme_seq != 0]
                    if len(phoneme_no_blank) >= 2:
                        P = mono_n_classes - 1
                        prev = phoneme_no_blank[:-1]
                        nxt = phoneme_no_blank[1:]
                        # diphone_id = (prev - 1) * P + (next - 1) + 1
                        diphone_true = (prev - 1) * P + (nxt - 1) + 1
                        diphone_true = diphone_true.tolist()
                    else:
                        diphone_true = []
                    
                    # DER: Use diphone predictions directly
                    diphone_pred = np.argmax(logits, axis=-1)  # (time,)
                    # CTC collapse: remove consecutive duplicates
                    diphone_pred = np.array([diphone_pred[0]] + 
                                          [diphone_pred[i] for i in range(1, len(diphone_pred)) 
                                           if diphone_pred[i] != diphone_pred[i-1]])
                    # Remove blanks (0)
                    diphone_pred = diphone_pred[diphone_pred != 0]
                    
                    all_diphone_preds.append(diphone_pred.tolist())
                    all_diphone_true.append(diphone_true)
                    
                    # PER: Marginalize diphone logits to phoneme logits
                    # Add all diphone probabilities that end with the same phoneme
                    phoneme_logits = marginalize_diphone_logits_vectorized(
                        logits[np.newaxis, :, :],  # Add batch dimension: (1, time, n_diphone_classes)
                        mono_n_classes=mono_n_classes,
                        use_log_space=True
                    )
                    phoneme_logits = phoneme_logits[0]  # Remove batch dimension: (time, n_phoneme_classes)
                    
                    # Get phoneme predictions
                    phoneme_pred = np.argmax(phoneme_logits, axis=-1)  # (time,)
                    # CTC collapse: remove consecutive duplicates
                    phoneme_pred = np.array([phoneme_pred[0]] + 
                                          [phoneme_pred[i] for i in range(1, len(phoneme_pred)) 
                                           if phoneme_pred[i] != phoneme_pred[i-1]])
                    # Remove blanks (0)
                    phoneme_pred = phoneme_pred[phoneme_pred != 0]
                    
                    # True phoneme sequence (from HDF5, already phonemes)
                    phoneme_true = phoneme_seq[phoneme_seq != 0].tolist()
                    
                    all_phoneme_preds.append(phoneme_pred.tolist())
                    all_phoneme_true.append(phoneme_true)
                else:
                    # Mono model: DER = PER
                    phoneme_pred = np.argmax(logits, axis=-1)
                    # CTC collapse: remove consecutive duplicates
                    phoneme_pred = np.array([phoneme_pred[0]] + 
                                          [phoneme_pred[i] for i in range(1, len(phoneme_pred)) 
                                           if phoneme_pred[i] != phoneme_pred[i-1]])
                    # Remove blanks (0)
                    phoneme_pred = phoneme_pred[phoneme_pred != 0]
                    
                    all_phoneme_preds.append(phoneme_pred.tolist())
                    all_phoneme_true.append(true_seq.tolist())
                    all_diphone_preds.append(phoneme_pred.tolist())  # Same for mono
                    all_diphone_true.append(true_seq.tolist())
                
                pbar.update(1)
    
    print()
    
    # Calculate metrics
    print("=" * 80)
    print("CALCULATING METRICS")
    print("=" * 80)
    
    # Calculate DER
    total_diphone_edit_distance = 0
    total_diphone_length = 0
    
    for pred, true in zip(all_diphone_preds, all_diphone_true):
        ed = calculate_edit_distance(pred, true)
        total_diphone_edit_distance += ed
        total_diphone_length += len(true)
    
    der = (total_diphone_edit_distance / total_diphone_length * 100) if total_diphone_length > 0 else 0.0
    
    # Calculate PER
    total_phoneme_edit_distance = 0
    total_phoneme_length = 0
    
    for pred, true in zip(all_phoneme_preds, all_phoneme_true):
        ed = calculate_edit_distance(pred, true)
        total_phoneme_edit_distance += ed
        total_phoneme_length += len(true)
    
    per = (total_phoneme_edit_distance / total_phoneme_length * 100) if total_phoneme_length > 0 else 0.0
    
    # Print results
    print("=" * 80)
    print("VALIDATION METRICS")
    print("=" * 80)
    print(f"Total validation trials: {total_val_trials}")
    print(f"Total diphone tokens: {total_diphone_length}")
    print(f"Total phoneme tokens: {total_phoneme_length}")
    print()
    print(f"DER (Diphone Error Rate): {der:.4f}%")
    print(f"PER (Phoneme Error Rate): {per:.4f}%")
    print("=" * 80)


if __name__ == '__main__':
    main()

