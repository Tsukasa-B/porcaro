# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/cfg/actuator_cfg.py
from dataclasses import MISSING
from isaaclab.utils import configclass

@configclass
class PamDelayModelCfg:
    """空気圧の遅れ要素（無駄時間＋一次遅れ）の設定"""
    delay_time: float = 0.04  # 無駄時間 [s] (配管長に依存)
    time_constant: float = 0.15  # 一次遅れ時定数 [s] (バルブ応答等)

@configclass
class PamHysteresisModelCfg:
    """PAMのヒステリシスモデル（簡易Prandtl-Ishlinskii等）の設定"""
    hysteresis_width: float = 0.1  # ヒステリシスの幅係数
    curve_shape_param: float = 2.0  # 曲率パラメータ

@configclass
class ActuatorNetModelCfg:
    """ActuatorNet (データ駆動モデル) の設定"""
    input_dim: int = 4  # 入力次元 (例: Pressure_cmd, Pressure_cur, Angle, Velocity)
    output_dim: int = 1  # 出力次元 (例: Torque or Force)
    hidden_units: list[int] = (64, 64)  # 中間層のユニット数
    model_path: str | None = None  # 学習済み重みファイルのパス (.pt)

@configclass
class PamGeometricCfg:
    """
    PAMの幾何学的特性および有効収縮率 (Effective Contraction Ratio) の設定
    """
    # True: 有効収縮率 (Slack補正あり) を使用 [新手法]
    # False: 単純な幾何学的収縮率 (Slack無視) を使用 [既存手法]
    enable_slack_compensation: bool = True

    # 各筋肉のワイヤー長さオフセット [m] (enable_slack_compensation=True の時のみ有効)
    # 正(+): たるみ (Slack) あり -> 力発生が遅れる (Sim-to-Realギャップの主因)
    # 負(-): 初期張力 (Pre-tension) あり -> 最初から力が発生
    # 順序: [DF(背屈), F(屈曲), G(握り)]
    wire_slack_offsets: tuple[float, ...] = (0.0, 0.0, 0.0)
    
    # 筋肉の自然長 L0 [m]
    natural_length: float = 0.150