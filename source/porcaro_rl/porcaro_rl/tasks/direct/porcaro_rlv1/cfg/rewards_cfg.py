# rewards_cfg.py
from __future__ import annotations
from isaaclab.utils import configclass

@configclass
class RewardsCfg:
    """報酬の重みを管理する設定 (Phase Matching Design)"""
    
    # --- リズム生成設定 (RhythmGenerator用) ---
    bpm_min: float = 60.0
    bpm_max: float = 160.0 # 少し難易度を上げる
    prob_rest: float = 0.3    # 30% 休符
    prob_double: float = 0.2  # 20% ダブル (残り50%はシングル)

    # --- 報酬重み (Total Reward Weights) ---
    # 軌道マッチング報酬 (タイミングと力が合っているか)
    weight_match: float = 50.0
    
    # 休符維持報酬 (休むべき時に休んでいるか / ペナルティの裏返し)
    # ※ ペナルティとして実装する場合は負の値ではなく、正の項として「静止度」を評価する手もあるが
    # ここでは「休符時の誤接触ペナルティ」の重みとする。
    weight_rest_penalty: float = 1.0

    # エネルギー効率 (無駄な力み抑制)
    weight_efficiency: float = 0.01

    # --- 評価基準パラメータ ---
    target_force_fd: float = 20.0 # 目標打撃力 [N]
    
    # マッチング報酬の許容誤差
    sigma_force: float = 5.0  # 力の許容幅
    sigma_time: float = 0.1   # 時間の許容幅 [s] (50msズレたら報酬激減)

    # 休符判定
    force_threshold_rest: float = 1.0 # これ以上の力が出たら休符失敗とみなす