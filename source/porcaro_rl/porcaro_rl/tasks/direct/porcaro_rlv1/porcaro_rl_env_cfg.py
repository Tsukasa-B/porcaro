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
from .cfg.actuator_cfg import PamDelayModelCfg, PamHysteresisModelCfg, ActuatorNetModelCfg, PamGeometricCfg
from .cfg.actuator_net_cfg import CascadedActuatorNetCfg


@configclass
class PorcaroRLEnvCfg(DirectRLEnvCfg):
    """Porcaro 環境用の設定クラス (基本構成 - Model B Default)"""
    
    # --- シミュレーション設定 ---
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,     # 5ms
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.4, dynamic_friction=0.4, restitution=0.5
        ),
    )

    # Dynamics Configurations
    pam_delay_cfg: PamDelayModelCfg | None = None
    pam_hysteresis_cfg: PamHysteresisModelCfg | None = None
    actuator_net_cfg: ActuatorNetModelCfg | None = None

    # --- [Geometric Defaults] ---
    # デフォルトは Model B (High-Fidelity) 相当の設定
    pam_geometric_cfg: PamGeometricCfg = PamGeometricCfg(
        model_type="B",
        enable_slack_compensation=True,
        enable_soft_engagement=True,
        use_abs_epsilon=False,
        viscosity=500.0,
        wire_slack_offsets=(0.00504, 0.02146, 0.01357),
        natural_length=0.150
    )

    # --- シーン・アセット・センサ設定 (変更なし) ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32, env_spacing=3.0, replicate_physics=True,
    )
    robot_cfg: ArticulationCfg = ROBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    drum_cfg:  RigidObjectCfg  = DRUM_CFG.replace(prim_path="/World/envs/env_.*/Drum")
    stick_contact_cfg: ContactSensorCfg = contact_forces_stick_at_drum_cfg
    drum_contact_cfg: ContactSensorCfg = drum_vs_stick_cfg
    
    # --- RL 設定 ---
    decimation: int = 4
    episode_length_s: float = 8.0
    action_space: int = 3
    observation_space: int = 30
    state_space: int = 0
    dof_names: list[str] = ["Base_link_Wrist_joint", "Hand_link_Grip_joint"]

    # --- その他 ---
    use_simple_rhythm: bool = True
    simple_rhythm_mode: str = "double"
    simple_rhythm_bpm: float = 160.0
    lookahead_horizon: float = 0.5

    # --- モジュール ---
    controller: TorqueControllerCfg = TorqueControllerCfg()
    logging: LoggingCfg = LoggingCfg()
    rewards: RewardsCfg = RewardsCfg() 
    reward_logging: RewardLoggingCfg = RewardLoggingCfg()
    events: PorcaroEventCfg | None = None


@configclass
class PorcaroEventCfg:
    """ドメインランダム化(DR)の設定コンテナ"""
    randomize_mass: EventTerm = None
    randomize_material: EventTerm = None
    reset_robot_joints: EventTerm = None

def apply_domain_randomization(cfg: PorcaroRLEnvCfg):
    # (既存の実装そのまま)
    cfg.events = PorcaroEventCfg()
    cfg.events.randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "mass_distribution_params": (0.8, 1.2), "operation": "scale"},
    )
    cfg.events.randomize_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "static_friction_range": (0.7, 1.3), "dynamic_friction_range": (0.7, 1.3), "restitution_range": (0.8, 1.0), "num_buckets": 64},
    )
    cfg.events.reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "position_range": (0.9, 1.1), "velocity_range": (-0.1, 0.1)},
    )


# =========================================================
#  実験条件ごとの設定クラス (Model A / B / C)
# =========================================================

# --- [Model A] Baseline: Modified IECON2025 ---
@configclass
class PorcaroRLEnvCfg_ModelA(PorcaroRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # 1. Pneumatic: Constant Tau, 1D Deadtime
        self.pam_delay_cfg = PamDelayModelCfg(
            model_type="A",  # <--- [New]
            max_delay_time=0.1,
            tau_const=0.15   # Baseline Time Constant
        )
        self.pam_hysteresis_cfg = None
        
        # 2. Geometric: No Viscosity, No Slack Comp, Abs Epsilon
        self.pam_geometric_cfg = PamGeometricCfg(
            model_type="A", # <--- [New]
            enable_slack_compensation=False, # オフセット無効
            enable_soft_engagement=False,    # 不感帯平滑化なし
            use_abs_epsilon=True,            # 絶対値使用 (たるみでも力発生)
            viscosity=0.0,                   # 粘性なし
            force_scale=1.0,
            wire_slack_offsets=(0.0, 0.0, 0.0) # 念のため0埋め
        )

        self.actuator_net_cfg = None

        # Controller: Ideal response (遅れはPamDelayModelで処理)
        self.controller.tau = 0.0
        self.controller.dead_time = 0.0
        self.controller.use_pressure_dependent_tau = False

@configclass
class PorcaroRLEnvCfg_ModelA_DR(PorcaroRLEnvCfg_ModelA):
    """Model A + DR"""
    def __post_init__(self):
        super().__post_init__()
        apply_domain_randomization(self)


# --- [Model B] Proposed: High-Fidelity ---
@configclass
class PorcaroRLEnvCfg_ModelB(PorcaroRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # 1. Pneumatic: 2D Pressure-dependent Tau & Deadtime
        self.pam_delay_cfg = PamDelayModelCfg(
            model_type="B", # <--- [New]
            max_delay_time=0.1
        )
        
        # 2. Geometric: Viscosity, Slack Comp, Soft Engagement
        self.pam_geometric_cfg = PamGeometricCfg(
            model_type="B", # <--- [New]
            enable_slack_compensation=True,
            enable_soft_engagement=True,
            use_abs_epsilon=False, # 負の収縮率を許容
            viscosity=500.0,
            force_scale=1.0,
            wire_slack_offsets=(0.00504, 0.02146, 0.01357) # Calibrated values
        )
        
        self.pam_hysteresis_cfg = PamHysteresisModelCfg(
            hysteresis_width=0.0854, 
            curve_shape_param=2.0
        )
        
        self.actuator_net_cfg = None

        self.controller.tau = 0.0
        self.controller.dead_time = 0.0
        self.controller.use_pressure_dependent_tau = False

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
        
        self.actuator_net_cfg = CascadedActuatorNetCfg(
            slack_offsets=(0.0, 0.0, 0.0) # ActuatorNetは自力で学習するためOffset不要か要検討
        )
        
        self.pam_delay_cfg = None
        self.pam_hysteresis_cfg = None
        # Geometric Cfgは ActuatorNet内部では使われないが、互換性のため残す

@configclass
class PorcaroRLEnvCfg_ModelC_DR(PorcaroRLEnvCfg_ModelC):
    """Model C + DR"""
    def __post_init__(self):
        super().__post_init__()
        apply_domain_randomization(self)