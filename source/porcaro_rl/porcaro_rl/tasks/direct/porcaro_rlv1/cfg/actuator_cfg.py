# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/cfg/actuator_cfg.py
from dataclasses import MISSING
from isaaclab.utils import configclass

@configclass
class PamDelayModelCfg:
    """空気圧の遅れ要素（無駄時間＋一次遅れ）の設定"""
    delay_time: float = 0.02  # 無駄時間 [s] (配管長に依存)
    time_constant: float = 0.05  # 一次遅れ時定数 [s] (バルブ応答等)

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