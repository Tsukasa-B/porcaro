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

    # --- ▼▼▼ 追加: 接触ペナルティ設定 ▼▼▼ ---
    # リズム外で接触していた場合のペナルティ（ステップごと）
    # ※ 50Hz駆動で1秒間触り続けると 50 * 0.5 = -25点 くらいの減点になるイメージ
    weight_contact_penalty: float = 0.5
    
    # ターゲット時刻の前後何秒間は「接触していてもペナルティなし」とするか
    # (例: 0.1秒なら、正解時刻の前後0.1秒間は触れていても減点されない)
    penalty_safe_window: float = 0.15
    # ----------------------------------------
    
    # --- 評価基準 ---
    target_force_fd: float = 20.0
    # 力の許容誤差 (標準偏差)
    sigma_f: float = 10.0
    # タイミングの許容誤差 [秒] (この秒数ズレたら報酬が約6割に減る目安) <-- 追加
    sigma_t: float = 0.05 
    
    # --- 打撃検出 ---
    hit_threshold_force: float = 1.0