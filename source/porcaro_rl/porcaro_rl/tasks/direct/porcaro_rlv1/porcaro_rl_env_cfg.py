# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/porcaro_rl_env_cfg.py
from __future__ import annotations
from pathlib import Path
import math
import torch
import os

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.schemas import MassPropertiesCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm

# Porcaro imports
from .cfg.assets import ROBOT_CFG, DRUM_CFG
from .cfg.sensors import contact_forces_stick_at_drum_cfg, drum_vs_stick_cfg
from .cfg.controller_cfg import TorqueControllerCfg
from .cfg.logging_cfg import LoggingCfg, RewardLoggingCfg
from .cfg.rewards_cfg import RewardsCfg
from .cfg.actuator_cfg import PamDelayModelCfg, PamHysteresisModelCfg, ActuatorNetModelCfg, PamGeometricCfg # <--- 追加


@configclass
class PorcaroRLEnvCfg(DirectRLEnvCfg):
    """Porcaro 環境用の設定クラス (基本構成 - DRなし)"""
    
    # --- シミュレーション設定 ---
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,     # 5ms
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.4, dynamic_friction=0.4, restitution=0.5
        ),
    )

    # ===================================================
    # PAM Dynamics 設定スロット (デフォルトはNone)
    # ===================================================
    pam_delay_cfg: PamDelayModelCfg | None = None
    pam_hysteresis_cfg: PamHysteresisModelCfg | None = None
    actuator_net_cfg: ActuatorNetModelCfg | None = None

    # --- シーン設定 ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32,
        env_spacing=3.0,
        replicate_physics=True,
    )
    
    # --- アセット設定 ---
    robot_cfg: ArticulationCfg = ROBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    drum_cfg:  RigidObjectCfg  = DRUM_CFG.replace(prim_path="/World/envs/env_.*/Drum")
    
    # --- センサ設定 ---
    stick_contact_cfg: ContactSensorCfg = contact_forces_stick_at_drum_cfg
    drum_contact_cfg: ContactSensorCfg = drum_vs_stick_cfg
    
    # --- RL 設定 ---
    decimation: int = 4 # 20ms (50Hz)
    episode_length_s: float = 8.0
    
    # --- 空間定義 ---
    action_space: int = 3
    observation_space: int = 30
    state_space: int = 0
    
    dof_names: list[str] = ["Base_link_Wrist_joint", "Hand_link_Grip_joint"]

    # --- 追加: 幾何学補正設定 ---
    # デフォルトは True (有効収縮率を使用) とし、オフセットは 0 (影響なし) で初期化
    pam_geometric_cfg: PamGeometricCfg = PamGeometricCfg(
        enable_slack_compensation=True,
        wire_slack_offsets=(0.001, 0.005, 0.005), # 後でキャリブレーション値をここに入れます 0.00xmmのワイヤーが正しく貼るまでの長さ
        natural_length=0.150
    )
    # --------------------------

    # ===================================================
    # ★ 追加箇所: シンプルリズム生成設定
    # ===================================================
    use_simple_rhythm: bool = True   # TrueにするとSimpleRhythmGeneratorを使用
    simple_rhythm_mode: str = "double" # "single", "double", "steady"
    simple_rhythm_bpm: float = 160.0    # steadyモード時のBPM

    # --- 追加設定 ---
    lookahead_horizon: float = 0.5

    # --- モジュール別設定 ---
    controller: TorqueControllerCfg = TorqueControllerCfg()
    logging: LoggingCfg = LoggingCfg()
    rewards: RewardsCfg = RewardsCfg() 
    reward_logging: RewardLoggingCfg = RewardLoggingCfg()
    
    # DR設定用のスロット (デフォルトはNone)
    events: PorcaroEventCfg | None = None


@configclass
class PorcaroEventCfg:
    """ドメインランダム化(DR)の設定コンテナ"""
    randomize_mass: EventTerm = None
    randomize_material: EventTerm = None
    reset_robot_joints: EventTerm = None


# =========================================================
#  DR設定を適用するヘルパー関数
# =========================================================
def apply_domain_randomization(cfg: PorcaroRLEnvCfg):
    """引数で渡されたcfgにDR設定を注入する"""
    cfg.events = PorcaroEventCfg()
    
    # 1. リンク質量のランダム化 (±20%)
    cfg.events.randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # 2. 物理マテリアル(摩擦・反発)のランダム化
    cfg.events.randomize_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.8, 1.0),
            "num_buckets": 64,
        },
    )
    
    # 3. リセット時の関節角度ノイズ
    cfg.events.reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.9, 1.1),
            "velocity_range": (-0.1, 0.1),
        },
    )


# =========================================================
#  実験条件ごとの設定クラス (Model A / B / C × DRあり/なし)
# =========================================================

# --- [Model A] 理想/簡易遅れモデル ---
@configclass
class PorcaroRLEnvCfg_ModelA(PorcaroRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.pam_delay_cfg = PamDelayModelCfg(delay_time=0.02, time_constant=0.05)
        self.pam_hysteresis_cfg = None
        self.actuator_net_cfg = None

@configclass
class PorcaroRLEnvCfg_ModelA_DR(PorcaroRLEnvCfg_ModelA):
    """Model A + DR"""
    def __post_init__(self):
        super().__post_init__()
        apply_domain_randomization(self)


# --- [Model B] ヒステリシスモデル ---
@configclass
class PorcaroRLEnvCfg_ModelB(PorcaroRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # 実機データに基づく遅れパラメータの微調整
        # cmdとmeasの乖離から、時定数を少し大きめ(0.05 -> 0.08)に見積もり
        self.pam_delay_cfg = PamDelayModelCfg(
            delay_time=0.02,      # むだ時間 (約20ms)
            time_constant=0.08    # 一次遅れ時定数 (実機の応答波形より調整)
        )
        
        # 分析に基づくヒステリシス設定
        self.pam_hysteresis_cfg = PamHysteresisModelCfg(
            hysteresis_width=0.2, # Unloading時に18%の出力低下
            curve_shape_param=2.0  # (現在の実装では未使用)
        )
        
        self.actuator_net_cfg = None

@configclass
class PorcaroRLEnvCfg_ModelB_DR(PorcaroRLEnvCfg_ModelB):
    """Model B + DR"""
    def __post_init__(self):
        super().__post_init__()
        apply_domain_randomization(self)


# --- [Model C] ActuatorNet ---
@configclass
class PorcaroRLEnvCfg_ModelC(PorcaroRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actuator_net_cfg = ActuatorNetModelCfg(
            input_dim=4, output_dim=1, hidden_units=[64, 64],
            # model_path="models/actuator_net.pt" 
        )
        self.pam_delay_cfg = None
        self.pam_hysteresis_cfg = None

@configclass
class PorcaroRLEnvCfg_ModelC_DR(PorcaroRLEnvCfg_ModelC):
    """Model C + DR"""
    def __post_init__(self):
        super().__post_init__()
        apply_domain_randomization(self)