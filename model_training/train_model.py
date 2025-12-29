# import argparse
# from omegaconf import OmegaConf
# from rnn_trainer import BrainToTextDecoder_Trainer

# # 로컬에서 사용법:
# #   mono:    python train_model.py
# #   mono:    python train_model.py rnn_args.yaml
# #   diphone: python train_model.py rnn_args_diphone.yaml

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     # --config 인자로 사용할 YAML 파일 경로를 받는다.
#     # 예: --config rnn_args_diphone.yaml
#     parser.add_argument(
#         "--config",
#         type=str,
#         default="rnn_args.yaml",   # 아무것도 안 주면 기존 mono 설정 사용
#     )

#     # SageMaker가 자동으로 추가하는 인자들(--model-dir 등)을 무시하기 위해 parse_known_args 사용
#     cli_args, _ = parser.parse_known_args()

#     cfg = OmegaConf.load(cli_args.config)
#     trainer = BrainToTextDecoder_Trainer(cfg)
#     metrics = trainer.train()
import os
import sys
import json
import s3fs

from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer

# Initialize s3fs for S3 file access
fs = s3fs.S3FileSystem(anon=False)


def get_config_path(default_path: str = "rnn_args.yaml") -> str:
    """
    어떤 설정 파일을 쓸지 결정한다.

    우선순위:
      1) SageMaker 환경변수 SM_HPS 에 들어있는 hyperparameters(config 키)
      2) 명령줄 인자 중 --config 다음에 오는 값
      3) 기본값(default_path)
    """
    # 1. SageMaker 가 넣어주는 hyperparameters 환경변수 사용
    hps_json = os.environ.get("SM_HPS")
    if hps_json:
        try:
            hps = json.loads(hps_json)
            cfg = hps.get("config")
            if cfg:
                print(f"[train_model] Using config from SM_HPS: {cfg}")
                return cfg
        except Exception as e:
            print(f"[train_model] Failed to parse SM_HPS: {e}")

    # 2. CLI 인자에서 --config 찾아보기 (로컬 실행용)
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            cfg = sys.argv[idx + 1]
            print(f"[train_model] Using config from CLI: {cfg}")
            return cfg

    # 3. 아무것도 없으면 기본값
    print(f"[train_model] Using default config: {default_path}")
    return default_path


if __name__ == "__main__":
    config_path = get_config_path("rnn_args.yaml")

    print(f"[train_model] Final config path: {config_path}")

    # Load config file (supports both local and S3 paths)
    if config_path.startswith('s3://'):
        with fs.open(config_path, 'rb') as f:
            cfg = OmegaConf.load(f)
    else:
        cfg = OmegaConf.load(config_path)
    
    # Merge SageMaker hyperparameters into config (override YAML values)
    # SageMaker passes hyperparameters as strings, so we need to convert them appropriately
    hps_json = os.environ.get("SM_HPS")
    if hps_json:
        try:
            hps = json.loads(hps_json)
            print(f"[train_model] Raw hyperparameters from SM_HPS: {hps}")
            print(f"[train_model] Merging hyperparameters: {list(hps.keys())}")
            for key, value in hps.items():
                if key == "config":
                    continue  # Skip config key, it's used to select the config file
                
                original_value = value
                # Convert string booleans to actual booleans
                if isinstance(value, str):
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif value.lower() == "none" or value.lower() == "null":
                        value = None
                
                # Set the value in config (will override YAML value)
                try:
                    # Use dictionary-style assignment for OmegaConf
                    cfg[key] = value
                    print(f"[train_model] Set {key} = {value} (was: {original_value}, type: {type(value).__name__})")
                except Exception as e:
                    print(f"[train_model] Failed to set {key} = {value}: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"[train_model] Failed to merge hyperparameters: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("[train_model] No SM_HPS environment variable found, skipping hyperparameter merge")
    
    # Log final config values for checkpoint loading
    print(f"[train_model] Final config - init_from_checkpoint: {cfg.get('init_from_checkpoint')} (type: {type(cfg.get('init_from_checkpoint')).__name__})")
    print(f"[train_model] Final config - init_checkpoint_path: {cfg.get('init_checkpoint_path')}")
    
    trainer = BrainToTextDecoder_Trainer(cfg)
    metrics = trainer.train()
