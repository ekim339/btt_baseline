"""
Ensemble Inference Script

This script loads multiple finetuned model checkpoints and ensembles their predictions
by averaging log probabilities before generating the final predicted sequence of phonemes.

For diphone models, it first marginalizes diphone probabilities to phoneme probabilities
before ensembling.
"""

import os
import torch
import numpy as np
import pandas as pd
import redis
from omegaconf import OmegaConf
import time
from tqdm import tqdm
import editdistance
import argparse
import s3fs
import h5py
from typing import List, Dict, Optional, Union

from rnn_model import GRUDecoder
from evaluate_model_helpers import *
from diphone_to_phoneme_marginalization import (
    marginalize_diphone_logits_to_phoneme_logits,
    marginalize_diphone_logits_vectorized
)

fs = s3fs.S3FileSystem(anon=False)

# --------------------------------------------------------------------------------
# Argument parser
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Ensemble multiple pretrained RNN models for inference.'
)
parser.add_argument(
    '--checkpoint_paths',
    type=str,
    nargs='+',
    required=True,
    help='List of S3 or local paths to model checkpoint directories. '
         'Each should contain checkpoint/args.yaml and checkpoint/best_checkpoint'
)
parser.add_argument(
    '--data_dir',
    type=str,
    default='s3://4k-woody-btt/4k/data/hdf5_data_final',
    help='S3 path to dataset directory.'
)
parser.add_argument(
    '--eval_type',
    type=str,
    default='test',
    choices=['val', 'test'],
    help='Evaluation type: "val" for validation set, "test" for test set.'
)
parser.add_argument(
    '--csv_path',
    type=str,
    default="s3://4k-woody-btt/4k/data/t15_copyTaskData_description.csv",
    help='Path to the CSV file with metadata about the dataset.'
)
parser.add_argument(
    '--gpu_number',
    type=int,
    default=0,
    help='GPU number to use for inference. -1 for CPU.'
)
parser.add_argument(
    '--output_dir',
    type=str,
    default=None,
    help='Directory to save output CSV file. Defaults to first checkpoint directory.'
)
parser.add_argument(
    '--use_language_model',
    action='store_true',
    help='Whether to use the language model for final predictions.'
)
parser.add_argument(
    '--ensemble_method',
    type=str,
    default='average_logits',
    choices=['average_logits', 'average_probs'],
    help='Ensemble method: "average_logits" (default) or "average_probs"'
)
args = parser.parse_args()


# --------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------
def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> tuple:
    """
    Load a model from a checkpoint directory.
    
    Args:
        checkpoint_path: Path to checkpoint directory (S3 or local)
        device: Device to load model on
    
    Returns:
        model: Loaded GRUDecoder model
        model_args: Model configuration (OmegaConf)
        is_diphone: Whether this is a diphone model
        mono_n_classes: Number of mono phoneme classes (if diphone model)
    """
    # Load model args
    args_path = os.path.join(checkpoint_path, 'checkpoint/args.yaml')
    if checkpoint_path.startswith('s3://'):
        with fs.open(args_path, 'rb') as f:
            model_args = OmegaConf.load(f)
    else:
        model_args = OmegaConf.load(args_path)
    
    # Determine if this is a diphone model
    n_classes = model_args['dataset']['n_classes']
    mono_n_classes = model_args['dataset'].get('mono_n_classes', None)
    
    # If n_classes is large (e.g., 1601) and mono_n_classes is specified, it's a diphone model
    is_diphone = False
    if mono_n_classes is not None:
        P = mono_n_classes - 1
        expected_diphone_classes = P * P + 1
        if n_classes == expected_diphone_classes:
            is_diphone = True
    elif n_classes > 100:  # Heuristic: mono models typically have ~41 classes
        # Try to infer mono_n_classes from n_classes
        # n_classes = P^2 + 1, so P = sqrt(n_classes - 1)
        P = int(np.sqrt(n_classes - 1))
        if P * P + 1 == n_classes:
            mono_n_classes = P + 1  # P + 1 (including BLANK)
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
    checkpoint_file = os.path.join(checkpoint_path, 'checkpoint/best_checkpoint')
    if checkpoint_path.startswith('s3://'):
        with fs.open(checkpoint_file, 'rb') as f:
            checkpoint = torch.load(f, map_location=device, weights_only=False)
    else:
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    
    # Rename keys to remove "module." prefix (from DataParallel)
    for key in list(checkpoint['model_state_dict'].keys()):
        if key.startswith("module."):
            checkpoint['model_state_dict'][key.replace("module.", "")] = checkpoint['model_state_dict'].pop(key)
        if key.startswith("_orig_mod."):
            checkpoint['model_state_dict'][key.replace("_orig_mod.", "")] = checkpoint['model_state_dict'].pop(key)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, model_args, is_diphone, mono_n_classes


def ensemble_logits(
    logits_list: List[np.ndarray],
    method: str = 'average_logits'
) -> np.ndarray:
    """
    Ensemble multiple logits by averaging.
    
    Args:
        logits_list: List of logits arrays, each of shape (batch, time, n_classes)
        method: 'average_logits' (average logits directly) or 'average_probs' (average probabilities)
    
    Returns:
        ensemble_logits: Averaged logits of shape (batch, time, n_classes)
    """
    if len(logits_list) == 0:
        raise ValueError("logits_list is empty")
    
    if len(logits_list) == 1:
        return logits_list[0]
    
    if method == 'average_logits':
        # Simple average of logits
        ensemble_logits = np.mean(logits_list, axis=0)
    elif method == 'average_probs':
        # Convert to probabilities, average, convert back to logits
        probs_list = [np.exp(logits - np.max(logits, axis=-1, keepdims=True)) for logits in logits_list]
        probs_list = [probs / np.sum(probs, axis=-1, keepdims=True) for probs in probs_list]
        avg_probs = np.mean(probs_list, axis=0)
        ensemble_logits = np.log(avg_probs + 1e-10)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    return ensemble_logits


# --------------------------------------------------------------------------------
# Main script
# --------------------------------------------------------------------------------
def main():
    checkpoint_paths = args.checkpoint_paths
    data_dir = args.data_dir
    eval_type = args.eval_type
    ensemble_method = args.ensemble_method
    
    print(f"Loading {len(checkpoint_paths)} models for ensembling...")
    print(f"Checkpoint paths: {checkpoint_paths}")
    print()
    
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
    
    # Load all models
    models = []
    model_configs = []
    is_diphone_flags = []
    mono_n_classes_list = []
    
    for checkpoint_path in checkpoint_paths:
        print(f"Loading model from {checkpoint_path}...")
        model, model_args, is_diphone, mono_n_classes = load_model_from_checkpoint(checkpoint_path, device)
        models.append(model)
        model_configs.append(model_args)
        is_diphone_flags.append(is_diphone)
        mono_n_classes_list.append(mono_n_classes)
        
        model_type = "diphone" if is_diphone else "mono"
        print(f"  Model type: {model_type}")
        if is_diphone:
            print(f"  Mono classes: {mono_n_classes}")
        print(f"  Total classes: {model_args['dataset']['n_classes']}")
        print()
    
    # Use the first model's config for data loading (assuming all models use same sessions)
    primary_config = model_configs[0]
    
    # Load CSV file
    if args.csv_path.startswith('s3://'):
        with fs.open(args.csv_path, 'rb') as f:
            b2txt_csv_df = pd.read_csv(f)
    else:
        b2txt_csv_df = pd.read_csv(args.csv_path)
    
    # Helper function to load h5py file (handles both S3 and local paths)
    def load_h5py_file_safe(file_path, csv_df):
        """Load h5py file, handling both S3 and local paths."""
        if file_path.startswith('s3://'):
            with fs.open(file_path, 'rb') as f:
                with h5py.File(f, 'r') as h5f:
                    return _load_data_from_h5py(h5f, csv_df)
        else:
            return load_h5py_file(file_path, csv_df)
    
    def _load_data_from_h5py(h5f, csv_df):
        """Extract data from an opened h5py file."""
        data = {
            'neural_features': [],
            'n_time_steps': [],
            'seq_class_ids': [],
            'seq_len': [],
            'transcriptions': [],
            'sentence_label': [],
            'session': [],
            'block_num': [],
            'trial_num': [],
            'corpus': [],
        }
        keys = list(h5f.keys())
        for key in keys:
            g = h5f[key]
            neural_features = g['input_features'][:]
            n_time_steps = g.attrs['n_time_steps']
            seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None
            seq_len = g.attrs['seq_len'] if 'seq_len' in g.attrs else None
            transcription = g['transcription'][:] if 'transcription' in g else None
            sentence_label = g.attrs['sentence_label'][:] if 'sentence_label' in g.attrs else None
            session_name = g.attrs['session']
            block_num = g.attrs['block_num']
            trial_num = g.attrs['trial_num']
            
            year, month, day = session_name.split('.')[1:]
            date = f'{year}-{month}-{day}'
            row = csv_df[(csv_df['Date'] == date) & (csv_df['Block number'] == block_num)]
            corpus_name = row['Corpus'].values[0]
            
            data['neural_features'].append(neural_features)
            data['n_time_steps'].append(n_time_steps)
            data['seq_class_ids'].append(seq_class_ids)
            data['seq_len'].append(seq_len)
            data['transcriptions'].append(transcription)
            data['sentence_label'].append(sentence_label)
            data['session'].append(session_name)
            data['block_num'].append(block_num)
            data['trial_num'].append(trial_num)
            data['corpus'].append(corpus_name)
        return data
    
    # Load data for each session
    test_data = {}
    total_test_trials = 0
    
    for session in primary_config['dataset']['sessions']:
        eval_file = os.path.join(data_dir, session, f'data_{eval_type}.hdf5')
        
        # Check if file exists
        if data_dir.startswith('s3://'):
            if fs.exists(eval_file):
                data = load_h5py_file_safe(eval_file, b2txt_csv_df)
                test_data[session] = data
                total_test_trials += len(test_data[session]["neural_features"])
                print(f'Loaded {len(test_data[session]["neural_features"])} {eval_type} trials for session {session}.')
        else:
            if os.path.exists(eval_file):
                data = load_h5py_file_safe(eval_file, b2txt_csv_df)
                test_data[session] = data
                total_test_trials += len(test_data[session]["neural_features"])
                print(f'Loaded {len(test_data[session]["neural_features"])} {eval_type} trials for session {session}.')
    
    print(f"Total number of {eval_type} trials: {total_test_trials}")
    print()
    
    # Run inference with all models
    print("Running inference with all models...")
    all_logits = {i: [] for i in range(len(models))}
    
    with tqdm(total=total_test_trials, desc='Predicting with models', unit='trial') as pbar:
        for session, data in test_data.items():
            input_layer = primary_config['dataset']['sessions'].index(session)
            
            for trial in range(len(data['neural_features'])):
                neural_input = data['neural_features'][trial]
                neural_input = np.expand_dims(neural_input, axis=0)
                neural_input = torch.tensor(neural_input, device=device, dtype=_amp_dtype())
                
                # Get logits from each model
                trial_logits = []
                for model_idx, (model, model_args) in enumerate(zip(models, model_configs)):
                    logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)
                    trial_logits.append(logits[0])  # Remove batch dimension
                    all_logits[model_idx].append(logits[0])
                
                pbar.update(1)
    
    print()
    
    # Marginalize diphone logits to phoneme logits if needed
    print("Marginalizing diphone logits to phoneme logits...")
    phoneme_logits_list = []
    target_n_classes = None
    
    for model_idx, (is_diphone, mono_n_classes) in enumerate(zip(is_diphone_flags, mono_n_classes_list)):
        if is_diphone:
            if mono_n_classes is None:
                raise ValueError(f"Model {model_idx + 1} is a diphone model but mono_n_classes is None")
            print(f"  Model {model_idx + 1}: Converting diphone logits to phoneme logits (mono_n_classes={mono_n_classes})...")
            marginalized_logits = []
            for logits in all_logits[model_idx]:
                # logits shape: (time, n_diphone_classes)
                phoneme_logits = marginalize_diphone_logits_vectorized(
                    logits,
                    mono_n_classes=mono_n_classes,
                    use_log_space=True
                )
                marginalized_logits.append(phoneme_logits)
            phoneme_logits_list.append(marginalized_logits)
            if target_n_classes is None:
                target_n_classes = mono_n_classes
            elif target_n_classes != mono_n_classes:
                raise ValueError(
                    f"Inconsistent mono_n_classes: Model {model_idx + 1} has {mono_n_classes}, "
                    f"but previous models have {target_n_classes}"
                )
        else:
            # Mono model - check n_classes
            n_classes = model_configs[model_idx]['dataset']['n_classes']
            print(f"  Model {model_idx + 1}: Already phoneme logits (n_classes={n_classes})...")
            phoneme_logits_list.append(all_logits[model_idx])
            if target_n_classes is None:
                target_n_classes = n_classes
            elif target_n_classes != n_classes:
                raise ValueError(
                    f"Inconsistent n_classes: Model {model_idx + 1} has {n_classes}, "
                    f"but previous models have {target_n_classes}"
                )
    
    print(f"All models will output {target_n_classes} phoneme classes after marginalization.")
    print()
    
    print()
    
    # Ensemble the logits by averaging log probabilities
    print(f"Ensembling phoneme logits using method: {ensemble_method}...")
    ensemble_logits_list = []
    
    for trial_idx in range(total_test_trials):
        # Collect phoneme logits from all models for this trial
        trial_logits = [phoneme_logits_list[model_idx][trial_idx] for model_idx in range(len(models))]
        
        # Convert to numpy if needed
        trial_logits = [logits if isinstance(logits, np.ndarray) else logits.cpu().numpy() 
                       for logits in trial_logits]
        
        # Ensemble: average log probabilities (log-space averaging)
        # This is equivalent to: log(mean(exp(logits))) but more numerically stable
        if ensemble_method == 'average_logits':
            # Simple average of logits (equivalent to geometric mean of probabilities)
            ensemble_logits_trial = np.mean(trial_logits, axis=0)
        elif ensemble_method == 'average_probs':
            # Convert to probabilities, average, convert back to logits
            # More numerically stable: subtract max before exp
            max_logits = np.max([np.max(logits) for logits in trial_logits])
            probs_list = [np.exp(logits - max_logits) for logits in trial_logits]
            # Normalize each probability distribution
            probs_list = [probs / (np.sum(probs, axis=-1, keepdims=True) + 1e-10) 
                         for probs in probs_list]
            avg_probs = np.mean(probs_list, axis=0)
            ensemble_logits_trial = np.log(avg_probs + 1e-10) + max_logits
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
        
        ensemble_logits_list.append(ensemble_logits_trial)
    
    print(f"Ensembled {len(ensemble_logits_list)} trials")
    print()
    
    # Store ensemble logits in test_data structure
    trial_idx = 0
    for session in test_data.keys():
        test_data[session]['ensemble_logits'] = []
        for _ in range(len(test_data[session]['neural_features'])):
            test_data[session]['ensemble_logits'].append(ensemble_logits_list[trial_idx])
            trial_idx += 1
    
    # Convert ensemble logits to phoneme sequences
    print("Generating phoneme sequences from ensemble logits...")
    all_predictions = []
    
    for session, data in test_data.items():
        data['pred_seq'] = []
        for trial in range(len(data['ensemble_logits'])):
            logits = data['ensemble_logits'][trial]  # Shape: (time, n_phoneme_classes)
            
            # Get most likely phoneme at each timestep
            pred_seq = np.argmax(logits, axis=-1)  # Shape: (time,)
            
            # CTC collapse: remove consecutive duplicates
            pred_seq_collapsed = [pred_seq[0]]
            for i in range(1, len(pred_seq)):
                if pred_seq[i] != pred_seq[i-1]:
                    pred_seq_collapsed.append(pred_seq[i])
            pred_seq = np.array(pred_seq_collapsed)
            
            # Remove blanks (class 0)
            pred_seq = pred_seq[pred_seq != 0]
            
            # Convert to phoneme strings
            pred_phonemes = [LOGIT_TO_PHONEME[int(p)] for p in pred_seq]
            data['pred_seq'].append(pred_phonemes)
            
            # Store for output
            all_predictions.append({
                'session': session,
                'block_num': data['block_num'][trial],
                'trial_num': data['trial_num'][trial],
                'predicted_phonemes': ' '.join(pred_phonemes),
                'predicted_phoneme_ids': pred_seq.tolist()
            })
            
            # Print some examples
            if trial < 3:
                print(f'Session: {session}, Block: {data["block_num"][trial]}, Trial: {data["trial_num"][trial]}')
                if eval_type == 'val' and 'seq_class_ids' in data:
                    sentence_label = data['sentence_label'][trial] if 'sentence_label' in data else None
                    true_seq = data['seq_class_ids'][trial][0:data['seq_len'][trial]]
                    true_phonemes = [LOGIT_TO_PHONEME[int(p)] for p in true_seq]
                    if sentence_label:
                        print(f'Sentence label:      {sentence_label}')
                    print(f'True phonemes:       {" ".join(true_phonemes)}')
                print(f'Predicted phonemes:  {" ".join(pred_phonemes)}')
                print()
    
    print(f"Generated phoneme sequences for {len(all_predictions)} test trials")
    print()
    
    # Save predictions to CSV
    output_dir = args.output_dir if args.output_dir else checkpoint_paths[0]
    output_file = os.path.join(
        output_dir,
        f'ensemble_{len(checkpoint_paths)}models_{eval_type}_phoneme_predictions_{time.strftime("%Y%m%d_%H%M%S")}.csv'
    )
    
    df_out = pd.DataFrame(all_predictions)
    
    if output_dir.startswith('s3://'):
        with fs.open(output_file, 'w') as f:
            df_out.to_csv(f, index=False)
    else:
        os.makedirs(output_dir, exist_ok=True)
        df_out.to_csv(output_file, index=False)
    
    print(f"Saved phoneme predictions to {output_file}")
    print()
    
    # Print summary statistics
    print("=" * 80)
    print("ENSEMBLE INFERENCE SUMMARY")
    print("=" * 80)
    print(f"Number of models: {len(checkpoint_paths)}")
    print(f"Ensemble method: {ensemble_method}")
    print(f"Total test trials: {total_test_trials}")
    print(f"Output file: {output_file}")
    print("=" * 80)
    
    # Optional: Language model inference (if requested)
    if args.use_language_model:
        print("Running language model inference on ensemble logits...")
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.flushall()
        
        remote_lm_input_stream = 'remote_lm_input'
        remote_lm_output_partial_stream = 'remote_lm_output_partial'
        remote_lm_output_final_stream = 'remote_lm_output_final'
        
        remote_lm_output_partial_lastEntrySeen = get_current_redis_time_ms(r)
        remote_lm_output_final_lastEntrySeen = get_current_redis_time_ms(r)
        remote_lm_done_resetting_lastEntrySeen = get_current_redis_time_ms(r)
        
        lm_results = []
        
        with tqdm(total=total_test_trials, desc='Running language model', unit='trial') as pbar:
            for session in test_data.keys():
                for trial in range(len(test_data[session]['ensemble_logits'])):
                    logits = rearrange_speech_logits_pt(test_data[session]['ensemble_logits'][trial])[0]
                    
                    remote_lm_done_resetting_lastEntrySeen = reset_remote_language_model(
                        r, remote_lm_done_resetting_lastEntrySeen
                    )
                    
                    remote_lm_output_partial_lastEntrySeen, decoded = send_logits_to_remote_lm(
                        r,
                        remote_lm_input_stream,
                        remote_lm_output_partial_stream,
                        remote_lm_output_partial_lastEntrySeen,
                        logits,
                    )
                    
                    remote_lm_output_final_lastEntrySeen, lm_out = finalize_remote_lm(
                        r,
                        remote_lm_output_final_stream,
                        remote_lm_output_final_lastEntrySeen,
                    )
                    
                    best_candidate_sentence = lm_out['candidate_sentences'][0]
                    
                    lm_results.append({
                        'session': session,
                        'block_num': test_data[session]['block_num'][trial],
                        'trial_num': test_data[session]['trial_num'][trial],
                        'predicted_sentence': best_candidate_sentence
                    })
                    
                    pbar.update(1)
        
        # Save language model results
        output_dir = args.output_dir if args.output_dir else checkpoint_paths[0]
        lm_output_file = os.path.join(
            output_dir,
            f'ensemble_{len(checkpoint_paths)}models_{eval_type}_lm_predictions_{time.strftime("%Y%m%d_%H%M%S")}.csv'
        )
        
        df_lm = pd.DataFrame(lm_results)
        
        if output_dir.startswith('s3://'):
            with fs.open(lm_output_file, 'w') as f:
                df_lm.to_csv(f, index=False)
        else:
            os.makedirs(output_dir, exist_ok=True)
            df_lm.to_csv(lm_output_file, index=False)
        
        print(f"Saved language model predictions to {lm_output_file}")
        print()


if __name__ == '__main__':
    main()

