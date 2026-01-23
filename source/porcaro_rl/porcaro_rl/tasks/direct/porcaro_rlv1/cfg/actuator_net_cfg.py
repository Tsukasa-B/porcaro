# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/cfg/actuator_net_cfg.py
from isaaclab.utils import configclass

@configclass
class PressureNetCfg:
    """NN1: 圧力ダイナミクス (Input -> P_est)"""
    # Input: [P_cmd, P_prev, epsilon_geo] = 3次元 (筋肉1本あたり)
    input_dim: int = 3   
    output_dim: int = 1  
    hidden_units: list[int] = (32, 32)
    activation: str = "relu"

@configclass
class ForceNetCfg:
    """NN2: 力生成 (Input -> Force)"""
    # Input: [P_est, eps_eff, eps_dot] = 3次元 (筋肉1本あたり)
    input_dim: int = 3
    output_dim: int = 1
    hidden_units: list[int] = (64, 64)
    activation: str = "relu"

@configclass
class CascadedActuatorNetCfg:
    """Model C: Physics-Informed Cascaded ActuatorNet 全体設定"""
    pressure_net: PressureNetCfg = PressureNetCfg()
    force_net: ForceNetCfg = ForceNetCfg()
    
    # --- Sim-to-Real 補正パラメータ ---
    # ワイヤーのたるみ/張り (DF, F, G) [m]
    slack_offsets: tuple[float, ...] = (0.0, 0.0, 0.0)
    
    # --- 幾何学パラメータ (Robot Design) ---
    # ※ controller_cfg.py と整合させる必要がありますが、学習モデルの独立性を保つためここにも定義します
    r: float = 0.014       # プーリー半径 [m]
    L0: float = 0.150      # 自然長 [m]
    # 基準角度 (DF, F, G) [deg] -> ラジアンに変換して使用
    theta_ref_deg: tuple[float, ...] = (0.0, -90.0, -45.0) 
    
    # 学習済みモデルのパス (.pt)
    ckpt_path: str | None = None