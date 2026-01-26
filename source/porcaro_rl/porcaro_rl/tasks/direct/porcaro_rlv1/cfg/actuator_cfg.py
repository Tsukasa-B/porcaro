# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/cfg/actuator_cfg.py
from dataclasses import MISSING
from isaaclab.utils import configclass

@configclass
class PamDelayModelCfg:
    """空気圧の遅れ要素（可変むだ時間＋可変一次遅れ）の設定"""
    
    # --- [1] 時定数 (Tau) の設定 ---
    # 圧力軸 [MPa]
    tau_pressure_axis: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    # 時定数値 [s] (pneumatic.py TAU_TAB)
    tau_values: tuple[float, ...] = (0.043, 0.045, 0.060, 0.066, 0.094, 0.131)

    # --- [2] むだ時間 (Deadtime/Lag) の設定 [復活] ---
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
    use_absolute_geometry: bool = False

@configclass
class PamModelA_GeometricCfg(PamGeometricCfg):
    """Model A用: 絶対値幾何学、スラックなし"""
    enable_slack_compensation: bool = False  # ワイヤーたるみ計算無効
    use_absolute_geometry: bool = True       # 符号無視・絶対値計算有効
    wire_slack_offsets: tuple[float, ...] = (0.0, 0.0, 0.0)

@configclass
class PamModelA_DynamicsCfg(PamDelayModelCfg):
    """Model A用: 固定時定数 + 1Dむだ時間"""
    # 2Dダイナミクス(ヒステリシス等)を無効化
    use_2d_dynamics: bool = False 
    
    # 固定時定数 (Model A定義: 0.09s)
    fixed_time_constant: float = 0.09
    
    # むだ時間は 1D Table (L(P)) を使用するフラグ
    use_1d_deadtime: bool = True