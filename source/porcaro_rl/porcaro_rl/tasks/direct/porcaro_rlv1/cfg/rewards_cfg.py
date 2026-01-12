# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/cfg/rewards_cfg.py
from __future__ import annotations
from isaaclab.utils import configclass

@configclass
class RewardsCfg:
    """報酬の重みを管理する設定 (Phase Matching Design)"""
    
    # --- リズム生成設定 ---
    bpm_min: float = 60.0
    bpm_max: float = 160.0
    prob_rest: float = 0.3
    prob_double: float = 0.2

    # --- 報酬重み (Total Reward Weights) ---
    
    # 1. マッチング報酬（強化）
    # 逃げるより叩いた方が遥かにお得だと思わせるため、50 -> 100 に倍増
    weight_match: float = 100.0
    
    # 2. 休符維持報酬（緩和）
    # 「絶対触るな」という圧を下げて、次の音の準備をしやすくする。1.0 -> 0.1
    weight_rest_penalty: float = 0.1

    # 以下の項目を RewardsCfg クラス内に追加してください
    # 1ステップあたりの常時接触ペナルティ (0.01sあたり)
    weight_contact_continuous: float = -2.0 
    # 許容される最大接触時間 (これを超えるとペナルティが激増する)
    max_contact_duration_s: float = 0.1

    # 3. 関節制限ペナルティ（★新規追加★）
    # 手首が-90度（天井）や+45度（床下）に行こうとしたら罰を与える
    weight_joint_limits: float = -10.0  # 強い負の報酬

    # エネルギー効率
    weight_efficiency: float = 0.01

    # --- 評価基準パラメータ ---
    target_force_fd: float = 20.0
    sigma_force: float = 5.0
    sigma_time: float = 0.1
    
    # ★新規追加: 関節角度のソフトリミット (deg)
    # この範囲を超えたら罰則発動
    # 手首(Wrist): -80度(上) 〜 +30度(下) 
    limit_wrist_range: tuple[float, float] = (-80.0, 30.0)
    # グリップ(Grip): 0度(開) 〜 60度(閉)
    limit_grip_range: tuple[float, float] = (0.0, 60.0)