# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/cfg/actuator_cfg.py
from dataclasses import MISSING
from isaaclab.utils import configclass

@configclass
class PamDelayModelCfg:
    """空気圧の遅れ要素（可変むだ時間＋可変一次遅れ）の設定"""
    
    # --- [New] Model Selection ---
    # "A": Baseline (Constant Tau, 1D Deadtime)
    # "B": Proposed (2D Pressure-dependent Tau & Deadtime)
    model_type: str = "B"  # <--- [New] 追加

    # --- [1] Model A Specific Parameters ---
    tau_const: float = 0.15  # Model Aで使用する固定時定数 [s]  # <--- [New] 追加

    # --- [2] Model B / Legacy Parameters ---
    # 圧力軸 [MPa]
    tau_pressure_axis: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    # 時定数値 [s] (pneumatic.py TAU_TAB)
    tau_values: tuple[float, ...] = (0.043, 0.045, 0.060, 0.066, 0.094, 0.131)

    # --- [3] むだ時間 (Deadtime/Lag) の設定 ---
    # 圧力軸 [MPa] (通常はTauと同じ軸を使うが、独立定義も可能にする)
    deadtime_pressure_axis: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    # むだ時間値 [s] (pneumatic.py L_TAB) -> 圧力が高いほど到達が早い
    deadtime_values: tuple[float, ...] = (0.038, 0.035, 0.032, 0.030, 0.023, 0.023)
    
    # 最大遅延バッファ確保用 (これ以上の遅延はクリップされる)
    max_delay_time: float = 0.1
    
    # 互換性維持のための古いパラメータ (LUT有効時は無視)
    delay_time: float = 0.04  
    time_constant: float = 0.15

@configclass
class PamHysteresisModelCfg:
    """PAMのヒステリシスモデル（簡易Prandtl-Ishlinskii等）の設定"""
    hysteresis_width: float = 0.1  # ヒステリシスの幅係数
    curve_shape_param: float = 2.0  # 曲率パラメータ

@configclass
class ActuatorNetModelCfg:
    """ActuatorNet (データ駆動モデル) の設定"""
    input_dim: int = 4
    output_dim: int = 1
    hidden_units: list[int] = (64, 64)
    model_path: str | None = None

@configclass
class PamGeometricCfg:
    """
    PAMの幾何学的特性および有効収縮率 (Effective Contraction Ratio) の設定
    """
    # [New] Model Type for Geometric Calculation
    model_type: str = "B"  # "A" or "B"  # <--- [New] 追加

    # --- Viscosity (Model B only) ---
    viscosity: float = 500.0  # 粘性係数 [N / (m/s)] (Model Aでは0として扱われる) # <--- [New] 追加

    # --- Engagement & Slack (Model B only) ---
    # True: 有効収縮率 (Slack補正あり) を使用 [新手法: Model B]
    # False: 単純な幾何学的収縮率 (Slack無視) を使用 [既存手法: Model A]
    enable_slack_compensation: bool = True  # Model BではTrue, AではFalse推奨

    # Soft Engagement (不感帯の平滑化) を使用するか
    enable_soft_engagement: bool = True        # <--- [New] 追加
    engagement_smoothness: float = 100.0         # <--- [New] 追加

    # --- Epsilon Calculation ---
    # True: Model A (abs使用 / 負の収縮率を絶対値として扱う)
    # False: Model B (負の収縮率を許容し、力ゼロとする)
    use_abs_epsilon: bool = False  # <--- [New] 追加

    # --- Force Map Scale ---
    force_scale: float = 1.0  # <--- [New] 追加

    # 各筋肉のワイヤー長さオフセット [m] (enable_slack_compensation=True の時のみ有効)
    # 順序: [DF(背屈), F(屈曲), G(握り)]
    wire_slack_offsets: tuple[float, ...] = (0.0, 0.0, 0.0)
    
    # 筋肉の自然長 L0 [m]
    natural_length: float = 0.150