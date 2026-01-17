# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/cfg/rewards_cfg.py
from __future__ import annotations
from isaaclab.utils import configclass

@configclass
class RewardsCfg:
    """報酬の重みと正規化オプション"""
    
    # --- 正規化オプション（★重要: これで最大値を一致させる） ---
    # Trueなら「ターゲット強度(20Nなど)」に比例して報酬が増える（フォルテシモ重視）
    # Falseなら「どんな強さでも」一致すれば満点は1.0（正確性重視）
    scale_reward_by_force_magnitude: bool = False 

    # --- 報酬重み (各項の r_i に掛かる係数 w_i) ---
    # 不要な項は 0.0 にすれば計算自体が無効化される設計にします

    # --- 打撃判定用パラメータ (新規追加) ---
    # 接触開始からこの時間(秒)以内のピークのみを「打撃」として評価する
    # これを超えてから力を込めても、ピーク値は更新されない（Pushing対策）
    impact_window_s: float = 0.04
    
    # 1. 打撃の一致度 (Hit Match)
    weight_match: float = 100.0
    
    # 2. 休符の遵守 (Rest Compliance) - 以前はPenaltyでしたが「守れば報酬」の方が安定する場合も
    weight_rest: float = 0.1   # 休符を守っている間、毎ステップ入る報酬
    weight_rest_penalty: float = -5.0 # 休符なのに触ってしまった時の罰

    # 3. 接触継続ペナルティ (Anti-Pushing)
    weight_contact_continuous: float = -1 
    max_contact_duration_s: float = 0.06

    # 4. 関節制限 (Joint Limits)
    weight_joint_limits: float = -1.0

    # 5. アクションの滑らかさ/省エネ
    weight_action_rate: float = 0.0
    weight_energy: float = 0.0

    # --- 評価基準パラメータ ---
    target_force_fd: float = 20.0 # 基準となる力
    sigma_force: float = 15.0      # 許容誤差の幅
    
    limit_wrist_range: tuple[float, float] = (-80.0, 30.0)