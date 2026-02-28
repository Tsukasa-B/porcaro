#!/bin/bash
# 1. ディレクトリと環境の保証
cd $(dirname "$0")
echo "Starting experimental pipeline at $(date)"

# Conda環境の有効化
source $(conda info --base)/etc/profile.d/conda.sh
conda activate env_isaaclab

# パス定義
BASE_CFG_DIR="source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/agents"
TARGET_CFG="${BASE_CFG_DIR}/rsl_rl_ppo_cfg.py"
MLP_SRC="${BASE_CFG_DIR}/rsl_rl_ppo_mlp_cfg.py"
LSTM_SRC="${BASE_CFG_DIR}/rsl_rl_ppo_lstm_cfg.py"

# --- Phase 1: MLP (午前2時開始) ---
echo "Waiting until 02:00 AM for Phase 1 (MLP)..."
while [ $(date +%H) -ne 02 ]; do
    sleep 60
done

echo "Setting up MLP configuration..."
cp "$MLP_SRC" "$TARGET_CFG"

echo "--- [02:00 AM] Starting Phase 1 (MLP) ---"
python scripts/rsl_rl/train.py --task Template-Porcaro-Direct-ModelB --num_envs 2048 --headless


# --- Phase 2: LSTM (午前6時開始) ---
echo "Waiting until 06:00 AM for Phase 2 (LSTM)..."
while [ $(date +%H) -lt 06 ]; do
    sleep 60
done

echo "Setting up LSTM configuration..."
cp "$LSTM_SRC" "$TARGET_CFG"

echo "--- [06:00 AM] Starting Phase 2 (LSTM) ---"
python scripts/rsl_rl/train.py --task Template-Porcaro-Direct-ModelB --num_envs 2048 --headless

echo "All tasks completed at $(date)"
