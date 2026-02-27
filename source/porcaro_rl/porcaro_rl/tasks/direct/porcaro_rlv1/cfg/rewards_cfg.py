# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/cfg/rewards_cfg.py
from __future__ import annotations
from isaaclab.utils import configclass

@configclass
class RewardsCfg:
    """報酬の重みと正規化オプション"""
    
    # --- 正規化オプション ---
    scale_reward_by_force_magnitude: bool = False 

    # --- ズル防止パラメータ ---
    # grip0のとき，手首が9度ぐらいで打面に接触するため、10度を「振りかぶった」と認めるギリギリの閾値とする
    swing_amplitude_threshold_deg: float = 0.5 

    # --- 報酬重み (w_i) ---

    # [Note]: 以下の *_s (秒数設定) は初期値です。
    # 実際にはBPMに基づいて動的に計算された値(Adaptive Thresholds)が使用されます。
    
    # 1. 打撃の一致度 (Hit Match)
    # 変更箇所: 10.0 -> 1.0
    weight_match: float = 1.0
    impact_window_s: float = 0.04 # (動的計算の基準として使用される場合あり)
    
    # 2. 休符の遵守 (Rest Compliance)
    # 変更箇所: Matchに合わせて1/10スケールダウン
    weight_rest: float = 0.005             # 変更: 0.1 -> 0.01
    weight_rest_penalty: float = -0.01    # 変更: -0.5 -> -0.05

    # 3. 接触継続ペナルティ (Anti-Pushing)
    # 変更箇所: 1/10スケールダウン
    weight_contact_continuous: float = -0.2  # 変更: -2.0 -> -0.2
    max_contact_duration_s: float = 0.04 

    # 4. その他
    weight_joint_limits: float = 0.0
    
    # 変更箇所: ミスペナルティを1/10スケールダウン
    weight_miss: float = -1.0             # 変更: -2.0 -> -0.2
    
    # 変更箇所: ダブル失敗時のペナルティをさらに緩和して1/10スケールダウン
    # （前回の提案であった「-0.5」のさらに1/10となる「-0.05」を設定します）
    weight_double_hit: float = -0.5      # 変更: -1.0 -> -0.05

    # =========================================================
    # 変更箇所: 身体性の創発を促すシンプルな制約（追加）
    # =========================================================
    # 手首の共収縮(ガチガチ)防止：DFとFが同時に高いとペナルティ
    weight_wrist_co_contract: float = -0.5
    
    # グリップの脱力促進：Gを強く握り続けるとペナルティ
    weight_grip_penalty: float = -0.5

    # --- 評価基準パラメータ ---
    target_force_fd: float = 20.0 # 基準となる力
    sigma_force: float = 15.0      # 許容誤差の幅
    
    limit_wrist_range: tuple[float, float] = (-100.0, 120.0)