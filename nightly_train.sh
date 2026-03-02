#!/bin/bash
# 1. ディレクトリと環境の保証
cd $(dirname "$0")
echo "Starting experimental pipeline at $(date)"

# Conda環境の有効化
source $(conda info --base)/etc/profile.d/conda.sh
conda activate env_isaaclab

# エージェント設定のパス定義
BASE_CFG_DIR="source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/agents"
TARGET_CFG="${BASE_CFG_DIR}/rsl_rl_ppo_cfg.py"
MLP_SRC="${BASE_CFG_DIR}/rsl_rl_ppo_mlp_cfg.py"
LSTM_SRC="${BASE_CFG_DIR}/rsl_rl_ppo_lstm_cfg.py"

# 変更箇所: 環境設定(Env Cfg)のパス定義を追加
BASE_ENV_CFG_DIR="source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1"
TARGET_ENV_CFG="${BASE_ENV_CFG_DIR}/porcaro_rl_env_cfg.py"
SRC_ENV_CFG="${BASE_ENV_CFG_DIR}/porcaro_rl_env_cfgv1.py"


# --- Phase 2: LSTM (午前6時開始) ---
echo "Waiting until 01:00 AM for Phase 2 (LSTM)..."
while [ $(date +%H) -lt 08 ]; do
    sleep 60
done

echo "Setting up LSTM & Environment configuration..."
cp "$LSTM_SRC" "$TARGET_CFG"
# 変更箇所: Phase 2開始前にも念のため環境設定をv1で上書き（手動で元に戻されていた場合の対策）
cp "$SRC_ENV_CFG" "$TARGET_ENV_CFG"

echo "--- [06:00 AM] Starting Phase 2 (LSTM) ---"
python scripts/rsl_rl/train.py --task Template-Porcaro-Direct-ModelB-DR --num_envs 2048 --headless

echo "All tasks completed at $(date)"