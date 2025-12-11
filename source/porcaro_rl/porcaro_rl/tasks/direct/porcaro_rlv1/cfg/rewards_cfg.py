# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/cfg/rewards_cfg.py

from __future__ import annotations
from isaaclab.utils import configclass

@configclass
class RewardsCfg:
    """報酬の重みを管理する設定"""
    
    # --- リズムタスク設定 ---
    target_bpm: float = 120.0
    
    # --- 報酬重み ---
    # r_F (打撃力) の重み
    weight_force: float = 10.0
    # r_T (タイミング) の重み <-- 追加
    weight_timing: float = 10.0
    
    # --- 評価基準 ---
    target_force_fd: float = 20.0
    # 力の許容誤差 (標準偏差)
    sigma_f: float = 10.0
    # タイミングの許容誤差 [秒] (この秒数ズレたら報酬が約6割に減る目安) <-- 追加
    sigma_t: float = 0.05 
    
    # --- 打撃検出 ---
    hit_threshold_force: float = 1.0