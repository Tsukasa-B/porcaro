# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/cfg/rewards_cfg.py

from __future__ import annotations
from isaaclab.utils import configclass

@configclass
class RewardsCfg:
    """報酬の重みを管理する設定"""
    
    # --- 既存の設定 ---
    target_force_fd: float = 20.0
    sigma_f: float = 10.0
    weight_w1_force: float = 10.0
    weight_w2_hit_count: float = 5.0
    hit_threshold_force: float = 1.0

    # --- ▼▼▼ 新規追加: リズム報酬の設定 ▼▼▼ ---
    
    # 目標BPM (Beats Per Minute)
    target_bpm: float = 120.0
    
    # タイミング報酬のガウス幅 sigma_t [秒]
    # 値が小さいほど、ジャストタイミング以外は評価されなくなる
    sigma_t: float = 0.05 
    
    # タイミング報酬の重み w3
    weight_w3_timing: float = 10.0