# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/cfg/rewards_cfg.py
from __future__ import annotations
from isaaclab.utils import configclass

@configclass
class RewardsCfg:
    """報酬の重みと正規化オプション"""
    
    # --- 正規化オプション ---
    scale_reward_by_force_magnitude: bool = False 

    # --- 報酬重み (w_i) ---

    # [Note]: 以下の *_s (秒数設定) は初期値です。
    # 実際にはBPMに基づいて動的に計算された値(Adaptive Thresholds)が使用されます。
    
    # 1. 打撃の一致度 (Hit Match)
    weight_match: float = 100.0
    impact_window_s: float = 0.04 # (動的計算の基準として使用される場合あり)
    
    # 2. 休符の遵守 (Rest Compliance)
    weight_rest: float = 0.5        # 時間積分項 (Scaleされるので少し大きめでもOK)
    weight_rest_penalty: float = -5.0

    # 3. 接触継続ペナルティ (Anti-Pushing)
    # これが重要。「速い曲なのに長く触っている」と強烈に罰せられます
    weight_contact_continuous: float = -50.0 
    max_contact_duration_s: float = 0.1 # (Adaptive Rewardにより上書きされます)

    # 4. その他
    weight_joint_limits: float = 0.0
    weight_miss: float = -20.0       # Match Scaleで増幅されるので適度な値に
    weight_double_hit: float = -10.0 # Match Scaleで増幅

    # --- 評価基準パラメータ ---
    target_force_fd: float = 20.0 # 基準となる力
    sigma_force: float = 15.0      # 許容誤差の幅
    
    limit_wrist_range: tuple[float, float] = (-100.0, 120.0)