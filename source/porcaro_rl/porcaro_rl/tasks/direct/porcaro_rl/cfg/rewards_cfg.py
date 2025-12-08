# rewards_cfg.py
from __future__ import annotations
from isaaclab.utils import configclass

@configclass
class RewardsCfg:
    """報酬の重みを管理する設定"""
    
    # --- 論文 III-B, III-C (P.4) に基づく設定 ---
    
    # r_F1: 目標打撃力 Fd [N]
    # (論文では 20, 40, 60 N で実験 [cite: 238])
    target_force_fd: float = 20.0
    
    # r_F1: ガウス関数の標準偏差 sigma_F [cite: 223, 238]
    sigma_f: float = 10.0
    
    # R: r_F1 (打撃力) の重み w1 [cite: 230]
    weight_w1_force: float = 10.0
    
    # R: r_h (打撃回数) の重み w2 [cite: 230]
    weight_w2_hit_count: float = 5.0
    
    # --- 打撃検出用 ---
    
    # 接触とみなす力の最小閾値 [N]
    # (論文では 1N 未満はノーヒット扱い [cite: 241])
    hit_threshold_force: float = 1.0