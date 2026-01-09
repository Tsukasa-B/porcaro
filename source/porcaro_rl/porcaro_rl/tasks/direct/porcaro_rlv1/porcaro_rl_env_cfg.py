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

import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
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
        dt=1 / 200,     # 物理エンジンが計算する最小単位1/200=5ms 
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.5
        ),
    )

    # --- シーン設定 (porcaro から移植) ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32, # ※ main() で上書きされます
        env_spacing=3.0,
        replicate_physics=True,
    )
    
    # --- アセット設定 (上記で定義した Cfg を使用) ---
    robot_cfg: ArticulationCfg = ROBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    drum_cfg:  RigidObjectCfg  = DRUM_CFG.replace(prim_path="/World/envs/env_.*/Drum")
    
    # --- センサ設定 (上記で定義した Cfg を使用) ---
    stick_contact_cfg: ContactSensorCfg = contact_forces_stick_at_drum_cfg
    drum_contact_cfg: ContactSensorCfg = drum_vs_stick_cfg
    
    # --- RL 設定 ---
    # dt=1/200, decimation=4 なので 制御周期=50Hz (20ms)　Agentの出力をdt*decimationとして計算する
    decimation: int = 4
    
    # ★変更: 1.0s -> 8.0s
    # 120BPM (0.5s間隔) で16回の打撃を行うのに十分な時間です。
    episode_length_s: float = 8.0
    
    # --- 空間定義 (porcaro の仕様に合わせる) ---
    # アクション空間: (theta_eq, K_wrist, K_grip) の 3次元
    action_space: int = 3
    # 観測空間: [関節位置(2), 関節速度(2)] + [リズム情報(4)] = 8次元に変更
    observation_space: int = 8  # <-- 変更: 4 -> 8
    state_space: int = 0
    
    # --- 関節名 (porcaro から移植) ---
    dof_names: list[str] = ["Base_link_Wrist_joint", "Hand_link_Grip_joint"]

    # --- モジュール別設定 (各 Cfg ファイルからインポート) ---
    controller: TorqueControllerCfg = TorqueControllerCfg()
    logging: LoggingCfg = LoggingCfg()
    rewards: RewardsCfg = RewardsCfg() 
    reward_logging: RewardLoggingCfg = RewardLoggingCfg()

@configclass
class PorcaroEventCfg:
    """DR設定をまとめるクラス"""
    pass


@configclass
class PorcaroRLEnvDRCfg(PorcaroRLEnvCfg):
    """
    [学習用] ドメインランダム化(DR)を追加した設定クラス。
    PorcaroRLEnvCfg を継承しているため、基本的な設定はすべて同じです。
    """
    # 2. ここで初期化（空の箱を用意）
    events: PorcaroEventCfg = PorcaroEventCfg()

    def __init__(self):
        super().__init__()

        self.events = PorcaroEventCfg()
        
        # === A. 物理パラメータのランダム化 ===
        
        # 1. リンク質量のランダム化 (±20%)
        # ロボットの製造誤差や配線重量の不確かさをカバーします
        self.events.randomize_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"), # ロボット全体
                "mass_distribution_params": (0.8, 1.2),
                "operation": "scale",
            },
        )

        # 2. 物理マテリアル(摩擦・反発)のランダム化
        # ドラムヘッドの跳ね返りやスティックの滑り具合の誤差をカバーします
        self.events.randomize_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "static_friction_range": (0.7, 1.3),  # 静止摩擦 ±30%
                "dynamic_friction_range": (0.7, 1.3), # 動摩擦 ±30%
                "restitution_range": (0.9, 1.1),      # 反発係数 ±10%
                "num_buckets": 64,
            },
        )
        
        # === B. 初期状態のランダム化 (必要に応じて調整) ===
        
        # 関節角度にノイズを乗せてリセットする
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "position_range": (0.9, 1.1), # デフォルト姿勢の90%~110%の範囲
                "velocity_range": (-0.1, 0.1), # わずかな初速度
            },
        )