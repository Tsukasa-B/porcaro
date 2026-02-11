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
from .cfg.actuator_cfg import (
    PamDelayModelCfg, ActuatorNetModelCfg, PamGeometricCfg,
    PamModelA_GeometricCfg, PamModelA_DynamicsCfg
)
from .cfg.actuator_net_cfg import CascadedActuatorNetCfg


@configclass
class PorcaroRLEnvCfg(DirectRLEnvCfg):
    """Porcaro 環境用の設定クラス (基本構成 - DRなし)"""

    # --- RL 設定 ---
    decimation: int = 4#10 
    
    # [変更]: 4小節構造 (BPM60時) に耐えられるよう十分な長さを確保
    # 実際のエピソード終了は BPM に基づき動的に判定されます
    episode_length_s: float = 20.0
    
    # --- シミュレーション設定 ---
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,#500,     # 1ms
        render_interval=decimation, # ここは絶対に変えてはいけない
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.0, dynamic_friction=0.0, restitution=0.0, # restitutionは０
        ),
    )

    # ===================================================
    # PAM Dynamics 設定スロット (デフォルトはNone)
    # Envでは直接使用しませんが、ActuatorNet構成の参照用に残します
    # ===================================================
    pam_delay_cfg: PamDelayModelCfg | None = None
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
    
    action_space: int = 3
    # q(2) + qd(2) + last_act(2) + phase(2) + bpm(1) + lookahead(25) = 34
    observation_space: int = 35
    state_space: int = 0
    
    dof_names: list[str] = ["Base_link_Wrist_joint", "Hand_link_Grip_joint"]

    # --- 幾何学補正設定 ---
    # デフォルトは True (有効収縮率を使用: Model B相当)
    pam_geometric_cfg: PamGeometricCfg = PamGeometricCfg(
        natural_length=0.150,
        use_absolute_geometry = False,
        wire_slack_offsets=(0.0, 0.0, 0.0),
    )

    # --- シンプルリズム生成設定 ---
    # [変更]: 統合された RhythmGenerator を使うため、モード指定等は維持しつつ
    # BPM範囲などを明示的に定義
    use_simple_rhythm: bool = False   
    simple_rhythm_mode: str = "single" 
    simple_rhythm_bpm: float = 10.0    
    target_hit_force: float = 30.0

    lookahead_horizon: float = 0.5
    
    # [追加]: ランダム時のBPM範囲
    bpm_range: tuple[float, float] = (60.0, 160.0)

    # デフォルトは (1.0, 1.0) でスケーリングなし（既存挙動維持）
    pam_tau_scale_range: tuple[float, float] = (1.0, 1.0)

    # --- モジュール別設定 ---
    controller: TorqueControllerCfg = TorqueControllerCfg()
    logging: LoggingCfg = LoggingCfg()
    rewards: RewardsCfg = RewardsCfg() 
    reward_logging: RewardLoggingCfg = RewardLoggingCfg()
    
    events: PorcaroEventCfg | None = None

    def __post_init__(self):
        super().__post_init__()
        if hasattr(self.rewards, "target_force_fd"):
            self.rewards.target_force_fd = self.target_hit_force


@configclass
class PorcaroEventCfg:
    """ドメインランダム化(DR)の設定コンテナ"""
    randomize_mass: EventTerm = None
    randomize_material: EventTerm = None
    reset_robot_joints: EventTerm = None


def apply_domain_randomization(cfg: PorcaroRLEnvCfg):
    cfg.events = PorcaroEventCfg()
    cfg.events.randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    cfg.events.randomize_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.75, 1.25),
            "dynamic_friction_range": (0.75, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    cfg.events.reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.95, 1.05),
            "velocity_range": (0, 0),
        },
    )


# =========================================================
#  実験条件ごとの設定クラス
# =========================================================

# --- [Model A] 理想/簡易遅れモデル ---
@configclass
class PorcaroRLEnvCfg_ModelA(PorcaroRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # [Dynamics] Envでは使いませんが、Model Aであることを明示
        self.pam_delay_cfg = None 
        
        # [Geometry] Model A専用設定 (絶対値, Slackなし)
        # TorqueActionControllerはこれを見て「Model Aだ」と判断します
        self.pam_geometric_cfg = PamModelA_GeometricCfg()
        self.actuator_net_cfg = None

        # Model Aの設定: 固定時定数を使用
        self.controller.tau = 0.09
        self.controller.dead_time = 0.0 # むだ時間はTable参照されるので0でOK
        self.controller.use_pressure_dependent_tau = True # これをTrueにするとTable Iがロードされる

@configclass
class PorcaroRLEnvCfg_ModelA_DR(PorcaroRLEnvCfg_ModelA):
    def __post_init__(self):
        super().__post_init__()
        apply_domain_randomization(self)


# --- [Model B] ヒステリシスモデル (Proposed) ---
@configclass
class PorcaroRLEnvCfg_ModelB(PorcaroRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.pam_delay_cfg = None
        self.actuator_net_cfg = None

        # Model Bの設定: 2D Dynamicsを使用
        # TorqueActionControllerは use_absolute_geometry=False を見て Model B と判断し、
        # 自動的に 2D Table をロードします。
        # ここでの tau は初期値としてのみ使われるため、0.09 (安全策) にしておきます。
        self.controller.tau = 0.09 
        self.controller.use_pressure_dependent_tau = False # 2D Mapを使うのでTable Iは不要
        
        self.controller.pam_viscosity = 0.0 

@configclass
class PorcaroRLEnvCfg_ModelB_DR(PorcaroRLEnvCfg_ModelB):
    def __post_init__(self):
        super().__post_init__()
        apply_domain_randomization(self)

        # 下限 0.3 (高速・高周波応答) 〜 上限 1.0 (低速・ステップ応答)
        self.pam_tau_scale_range = (0.3, 1.0)


# --- [Model C] ActuatorNet ---
@configclass
class PorcaroRLEnvCfg_ModelC(PorcaroRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.actuator_net_cfg = CascadedActuatorNetCfg(
            slack_offsets=(0.0, 0.0, 0.0)
        )
        self.pam_delay_cfg = None
        self.pam_hysteresis_cfg = None

@configclass
class PorcaroRLEnvCfg_ModelC_DR(PorcaroRLEnvCfg_ModelC):
    def __post_init__(self):
        super().__post_init__()
        apply_domain_randomization(self)