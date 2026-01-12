# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/porcaro_rl_env.py
from __future__ import annotations
import argparse
import torch
from collections.abc import Sequence

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.app import AppLauncher
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import ContactSensor
from isaaclab.managers import EventManager

# Porcaro RL imports
from .porcaro_rl_env_cfg import PorcaroRLEnvCfg
from .actions.base import ActionController
from .cfg.actuator_cfg import PamDelayModelCfg, PamHysteresisModelCfg, ActuatorNetModelCfg
from .actions.pam_dynamics import PamDelayModel, PamHysteresisModel, ActuatorNetModel
from .actions.torque import TorqueActionController
from .logging.logging_manager import LoggingManager
from .rewards.reward import RewardManager
from .rhythm_generator import RhythmGenerator


class PorcaroRLEnv(DirectRLEnv):
    """Porcaro 環境クラス (ActuatorNet & Sim-to-Real対応版)"""

    cfg: PorcaroRLEnvCfg

    def __init__(self, cfg: PorcaroRLEnvCfg, render_mode: str | None = None, **kwargs):
        # アセット/センサ用の変数を初期化
        self.robot: Articulation = None
        self.drum: RigidObject = None
        self.stick_sensor: ContactSensor = None
        self.drum_sensor: ContactSensor = None

        # コントローラ・マネージャ関連
        self.action_controller: ActionController = None
        self.logging_manager: LoggingManager = None
        self.reward_manager: RewardManager = None
        
        # ActuatorNet / Dynamics モデル（Noneで初期化）
        self.pam_delay: PamDelayModel | None = None
        self.pam_hysteresis: PamHysteresisModel | None = None
        self.actuator_net: ActuatorNetModel | None = None

        # 親クラスの __init__ を呼ぶ
        super().__init__(cfg, render_mode, **kwargs)

        # ---------------------------------------------------------
        # 1. 関節インデックスの特定
        # ---------------------------------------------------------
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        
        if len(self.dof_idx) != 2:
            raise ValueError(f"Expected 2 DOFs (wrist, grip), but found {len(self.dof_idx)} based on {self.cfg.dof_names}")
        self.joint_ids_tuple: tuple[int, int] = (self.dof_idx[0], self.dof_idx[1])
        
        # アクションバッファの初期化
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # ---------------------------------------------------------
        # 2. アクションコントローラの初期化
        # ---------------------------------------------------------
        dt_ctrl = self.cfg.sim.dt
        ctrl_cfg = self.cfg.controller
        
        self.action_controller = TorqueActionController(
            dt_ctrl=dt_ctrl,
            control_mode=ctrl_cfg.control_mode,
            r=ctrl_cfg.r,
            L=ctrl_cfg.L,
            theta_t_DF_deg=ctrl_cfg.theta_t_DF_deg,
            theta_t_F_deg=ctrl_cfg.theta_t_F_deg,
            theta_t_G_deg=ctrl_cfg.theta_t_G_deg,
            Pmax=ctrl_cfg.Pmax,
            tau=ctrl_cfg.tau,
            dead_time=ctrl_cfg.dead_time,
            N=ctrl_cfg.N,
            force_map_csv=ctrl_cfg.force_map_csv,
            h0_map_csv=ctrl_cfg.h0_map_csv,
            use_pressure_dependent_tau=ctrl_cfg.use_pressure_dependent_tau,
        )
        self.action_controller.reset(self.num_envs, self.device)

        # ★★★ ActuatorNet / PAM Dynamics の初期化 ★★★
        # ここではインスタンス化だけ行い、reset()は呼びません(forwardでの遅延初期化に任せます)
        
        # A. 遅れモデル
        if hasattr(self.cfg, "pam_delay_cfg") and self.cfg.pam_delay_cfg is not None:
             self.pam_delay = PamDelayModel(self.cfg.pam_delay_cfg, dt_ctrl, self.device)
        
        # B. ヒステリシスモデル
        if hasattr(self.cfg, "pam_hysteresis_cfg") and self.cfg.pam_hysteresis_cfg is not None:
             self.pam_hysteresis = PamHysteresisModel(self.cfg.pam_hysteresis_cfg, self.device)

        # C. ActuatorNet
        if hasattr(self.cfg, "actuator_net_cfg") and self.cfg.actuator_net_cfg is not None:
             self.actuator_net = ActuatorNetModel(self.cfg.actuator_net_cfg, self.device)
             
        # ---------------------------------------------------------
        # 3. 各種マネージャの初期化
        # ---------------------------------------------------------
        self.logging_manager = LoggingManager(
            env=self,
            dt=self.cfg.sim.dt,
            log_filepath=self.cfg.logging.filepath, 
            enable_logging=self.cfg.logging.enabled
        )

        self.reward_manager = RewardManager(
            cfg=self.cfg.rewards,
            num_envs=self.num_envs,
            device=self.device,
        )

        # リズム生成器
        bpm_range = (
            getattr(self.cfg.rewards, "bpm_min", 60.0),
            getattr(self.cfg.rewards, "bpm_max", 160.0)
        )
        prob_rest = getattr(self.cfg.rewards, "prob_rest", 0.3)
        prob_double = getattr(self.cfg.rewards, "prob_double", 0.2)
        self.dt_ctrl_step = self.cfg.sim.dt * self.cfg.decimation

        self.rhythm_generator = RhythmGenerator(
            num_envs=self.num_envs,
            device=self.device,
            dt=self.dt_ctrl_step,
            max_episode_length=self.max_episode_length,
            bpm_range=bpm_range,
            prob_rest=prob_rest,
            prob_double=prob_double
        )
        
        lookahead_horizon = getattr(self.cfg, "lookahead_horizon", 0.5)
        self.lookahead_steps = int(lookahead_horizon / self.dt_ctrl_step)
        
        # ---------------------------------------------------------
        # 4. 診断情報の表示
        # ---------------------------------------------------------
        print("=" * 80)
        print(f"[DEBUG DIAGNOSTIC] 環境設定値の確認")
        print(f"  - sim.dt (物理ステップ): {self.cfg.sim.dt}")
        print(f"  - decimation (間引き数): {self.cfg.decimation}")
        print(f"  - dt_ctrl (制御周期)   : {self.dt_ctrl_step:.5f} 秒")
        print("=" * 80)

        if self.cfg.events is not None:
            self.event_manager = EventManager(self.cfg.events, self)
            self.event_manager.apply(mode="startup")
        else:
            self.event_manager = None

        all_ids = torch.arange(self.num_envs, device=self.device)
        self.rhythm_generator.reset(all_ids)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.drum = RigidObject(self.cfg.drum_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["drum"] = self.drum
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.stick_sensor = ContactSensor(self.cfg.stick_contact_cfg)
        self.drum_sensor = ContactSensor(self.cfg.drum_contact_cfg)
        self.scene.sensors["stick_contact"] = self.stick_sensor
        self.scene.sensors["drum_contact"] = self.drum_sensor

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, rew, terminated, truncated, extras = super().step(actions)

        if hasattr(self, "common_step_counter"):
            TOTAL_TRAINING_STEPS = 100_000
            difficulty = min(self.common_step_counter / TOTAL_TRAINING_STEPS, 1.0)
            if hasattr(self, "rhythm_generator"):
                self.rhythm_generator.set_difficulty(difficulty)
            extras["Episode/difficulty"] = torch.tensor(difficulty, device=self.device)

        return obs, rew, terminated, truncated, extras

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions is not None:
            self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        """アクションの適用 (モデルの切り替えロジックを含む)"""
        
        # 1. 基本のアクション (Neural Networkの出力: -1 ~ 1)
        raw_cmd = (self.actions + 1.0) / 2.0
        
        # 2. ActuatorNet (Model C)
        if self.actuator_net is not None:
             # ※ 現在は枠組みのみ。入力は要調整
             pass 

        # 3. 簡易ヒステリシスモデル (Model B)
        if self.pam_hysteresis is not None:
            raw_cmd = self.pam_hysteresis(raw_cmd)
        
        # 4. 遅れモデル (Model A/B)
        if self.pam_delay is not None:
            raw_cmd = self.pam_delay(raw_cmd)

        # 5. 指令値をコントローラ用に戻す
        processed_actions = (raw_cmd * 2.0) - 1.0
        processed_actions = torch.clamp(processed_actions, -1.0, 1.0)

        # 6. トルク計算・適用
        q_full = self.robot.data.joint_pos 
        self.action_controller.apply(
            actions=processed_actions,
            q=q_full,
            joint_ids=self.joint_ids_tuple,
            robot=self.robot
        )

        # 7. ログ収集
        if self.logging_manager.enable_logging:
            current_steps = self.episode_length_buf
            tgt_val = self.rhythm_generator.get_current_target(current_steps)[0].item()
            tgt_bpm = self.rhythm_generator.current_bpms[0].item()

            self.logging_manager.update_time(self.cfg.sim.dt)
            self.logging_manager.buffer_step_data(
                q_full=self.robot.data.joint_pos,
                qd_full=self.robot.data.joint_vel,
                telemetry=self.action_controller.get_last_telemetry(),
                actions=self.actions,
                current_sim_time=self.sim.current_time,
                target_force=tgt_val,
                target_bpm=tgt_bpm
            )

    def _get_observations(self) -> dict:
        q = self.robot.data.joint_pos[:, self.dof_idx]
        qd = self.robot.data.joint_vel[:, self.dof_idx]
        
        current_steps = self.episode_length_buf
        bpm_obs = self.rhythm_generator.current_bpms.view(self.num_envs, 1) / 180.0
        rhythm_buf = self.rhythm_generator.get_lookahead(current_steps, self.lookahead_steps)
        target_force = getattr(self.cfg.rewards, "target_force_fd", 20.0)
        rhythm_buf = rhythm_buf / target_force

        obs = torch.cat((q, qd, bpm_obs, rhythm_buf), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        force_history_tensor = self.stick_sensor.data.net_forces_w_history
        current_rl_step_force_history = force_history_tensor[:, 0:self.cfg.decimation, :]
        force_z_history_in_step = current_rl_step_force_history[..., 2]
        current_force_max = torch.max(force_z_history_in_step, dim=1).values.clamp(min=0.0)
        
        current_steps = self.episode_length_buf
        target_val = self.rhythm_generator.get_current_target(current_steps)
        
        target_val = target_val.view(-1)
        current_force_max = current_force_max.view(-1)
        
        is_hit_target = target_val > 1.0
        force_error = current_force_max - target_val
        sigma_force = getattr(self.cfg.rewards, "sigma_force", 10.0)
        
        rew_match = torch.exp(- (force_error**2) / (sigma_force**2))
        rew_match = torch.where(is_hit_target, rew_match, torch.zeros_like(rew_match))
        
        force_threshold_rest = getattr(self.cfg.rewards, "force_threshold_rest", 1.0)
        is_rest_target = target_val <= 1.0
        violation = (current_force_max > force_threshold_rest)
        
        rew_rest_penalty = torch.where(
            is_rest_target & violation,
            -1.0 * current_force_max, 
            torch.zeros_like(rew_match)
        )

        weight_match = getattr(self.cfg.rewards, "weight_match", 20.0)
        weight_rest = getattr(self.cfg.rewards, "weight_rest_penalty", 10.0)
        total_reward = (weight_match * rew_match) + (weight_rest * rew_rest_penalty)

        self.reset_time_outs = self.episode_length_buf >= (self.max_episode_length - 1)
        self.reset_terminated = torch.zeros_like(self.reset_time_outs)
        
        if self.logging_manager.enable_logging:
            f1_dummy = torch.zeros_like(total_reward)
            self.logging_manager.finalize_log_step(
                peak_force=force_history_tensor,
                f1_force=f1_dummy,
                step_reward=total_reward
            )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.reset_terminated, self.reset_time_outs

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        if hasattr(self, "event_manager") and self.event_manager is not None:
            self.event_manager.reset(env_ids)
        if hasattr(self, "reward_manager"):
            self.reward_manager.reset_idx(env_ids)
        if hasattr(self, "logging_manager"):
            self.logging_manager.reset_idx(env_ids)
        if hasattr(self, "rhythm_generator"):
            self.rhythm_generator.reset(env_ids)
            
        # ★★★ ActuatorNet/PAMモデルのリセット (reset_idxを使用) ★★★
        if self.pam_delay is not None:
            self.pam_delay.reset_idx(env_ids)
        if self.pam_hysteresis is not None:
            self.pam_hysteresis.reset_idx(env_ids)

    def close(self):
        if hasattr(self, "logging_manager") and self.logging_manager is not None:
            self.logging_manager.save_on_exit()
        super().close()


# 実行用 main 関数
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true", help="UI なしで実行")
    p.add_argument("--num_envs", type=int, default=128, help="並列環境数")
    p.add_argument("--max_steps", type=int, default=1000, help="実行ステップ数")
    return p.parse_args()

def main():
    args = _parse_args()
    app = AppLauncher(headless=args.headless).app
    try:
        cfg = PorcaroRLEnvCfg()
        cfg.scene.num_envs = args.num_envs
        env = PorcaroRLEnv(cfg)
        print("[INFO] 環境をリセットします...")
        _ = env.reset()
        print(f"[INFO] {args.max_steps} ステップのシミュレーションを実行します。")
        for i in range(args.max_steps):
            actions = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
            obs, rew, terminated, truncated, info = env.step(actions)
            if (terminated | truncated).any():
                env.reset_done(terminated | truncated)
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals() and hasattr(env, 'logging_manager') and env.logging_manager:
            env.logging_manager.save_on_exit()
        app.close()

if __name__ == "__main__":
    main()