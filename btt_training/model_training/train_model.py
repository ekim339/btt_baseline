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

from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer


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

    cfg = OmegaConf.load(config_path)
    trainer = BrainToTextDecoder_Trainer(cfg)
    metrics = trainer.train()
