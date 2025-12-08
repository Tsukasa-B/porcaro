# porcaro_env_cfg.py
from __future__ import annotations
from pathlib import Path
import math
import torch
import os

# --- porcaro_rl_env_cfg.py から移植 (アセット定義) ---
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.schemas import MassPropertiesCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .cfg.assets import ROBOT_CFG, DRUM_CFG
from .cfg.sensors import contact_forces_stick_at_drum_cfg, drum_vs_stick_cfg
from .cfg.controller_cfg import TorqueControllerCfg
from .cfg.logging_cfg import LoggingCfg, RewardLoggingCfg
from .cfg.rewards_cfg import RewardsCfg


@configclass
class PorcaroRLEnvCfg(DirectRLEnvCfg):
    """Porcaro 環境用の設定クラス (チュートリアルベース)"""
    
    # --- シミュレーション設定 (porcaro から移植) ---
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200, 
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.5
        ),
    )

    # --- シーン設定 (porcaro から移植) ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512, # ※ main() で上書きされます
        env_spacing=3.0,
        replicate_physics=True,
    )
    
    # --- アセット設定 (上記で定義した Cfg を使用) ---
    robot_cfg: ArticulationCfg = ROBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    drum_cfg:  RigidObjectCfg  = DRUM_CFG.replace(prim_path="/World/envs/env_.*/Drum")
    
    # --- センサ設定 (上記で定義した Cfg を使用) ---
    stick_contact_cfg: ContactSensorCfg = contact_forces_stick_at_drum_cfg
    drum_contact_cfg: ContactSensorCfg = drum_vs_stick_cfg
    
    # --- RL 設定 (porcaro から移植 / 調整) ---
    decimation: int = 4
    episode_length_s: float = 1.0
    
    # --- 空間定義 (porcaro の仕様に合わせる) ---
    # アクション空間: wrist, grip の2関節トルク
    action_space: int = 3
    # 観測空間: wrist, grip の [関節位置(2), 関節速度(2)] = 4次元
    # (※ porcaro_env.py の _get_observations 実装に合わせる)
    observation_space: int = 4
    state_space: int = 0
    
    # --- 関節名 (porcaro から移植) ---
    dof_names: list[str] = ["Base_link_Wrist_joint", "Hand_link_Grip_joint"]

    # --- モジュール別設定 (各 Cfg ファイルからインポート) ---
    controller: TorqueControllerCfg = TorqueControllerCfg()
    logging: LoggingCfg = LoggingCfg()
    rewards: RewardsCfg = RewardsCfg() 
    reward_logging: RewardLoggingCfg = RewardLoggingCfg()