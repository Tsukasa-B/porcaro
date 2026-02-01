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
from .actions.pam_dynamics import ActuatorNetModel
from .actions.torque import TorqueActionController
from .logging.logging_manager import LoggingManager
from .rewards.reward import RewardManager
from .rhythm_generator import RhythmGenerator
from .simple_rhythm_generator import SimpleRhythmGenerator
from .cfg.assets import WRIST_J0, GRIP_J0
from .cfg.actuator_net_cfg import CascadedActuatorNetCfg # <--- 追加
from .actions.actuator_net import CascadedActuatorNet


class PorcaroRLEnv(DirectRLEnv):
    """Porcaro 環境クラス (ActuatorNet & Sim-to-Real対応版 - Fixed Logging)"""

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

        # =========================================================
        # ★ [維持] サブステップ間の最大力を保持するバッファ
        # =========================================================
        self.max_force_z_buffer = torch.zeros(self.num_envs, device=self.device)

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
            force_scale=ctrl_cfg.force_scale,
            h0_map_csv=ctrl_cfg.h0_map_csv,
            use_pressure_dependent_tau=ctrl_cfg.use_pressure_dependent_tau,
            geometric_cfg=self.cfg.pam_geometric_cfg,
        )
        self.action_controller.reset(self.num_envs, self.device)


        


        # --- [修正] ActuatorNet の分岐初期化 ---
        if hasattr(self.cfg, "actuator_net_cfg") and self.cfg.actuator_net_cfg is not None:
             if isinstance(self.cfg.actuator_net_cfg, CascadedActuatorNetCfg):
                 # Model C (New)
                 self.actuator_net = CascadedActuatorNet(self.cfg.actuator_net_cfg, self.device)
             else:
                 # Model B (Old / Simple MLP)
                 self.actuator_net = ActuatorNetModel(self.cfg.actuator_net_cfg, self.device)

        # Model C用の状態変数 (前回の推定内圧) [Batch, 3] (DF, F, G)
        self.last_pressure_est = torch.zeros((self.num_envs, 3), device=self.device)
             
        # ---------------------------------------------------------
        # 3. 各種マネージャの初期化
        # ---------------------------------------------------------
        print("=" * 80)
        print(f"[INFO] Logging Configuration Check:")
        print(f"  - Enabled : {self.cfg.logging.enabled}")
        print(f"  - Filepath: {self.cfg.logging.filepath}")
        print("=" * 80)

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

        # =========================================================
        # リズム生成器の切り替えロジック (維持)
        # =========================================================
        self.dt_ctrl_step = self.cfg.sim.dt * self.cfg.decimation
        use_simple = getattr(self.cfg, "use_simple_rhythm", False)
        target_force_val = getattr(self.cfg, "target_hit_force", 50.0)

        if use_simple:
            mode = getattr(self.cfg, "simple_rhythm_mode", "single")
            bpm = getattr(self.cfg, "simple_rhythm_bpm", 60.0)
            print(f"[INFO] SimpleRhythmGenerator (Mode: {mode}, BPM: {bpm}, Force: {target_force_val})")
            self.rhythm_generator = SimpleRhythmGenerator(
                num_envs=self.num_envs,
                device=self.device,
                dt=self.dt_ctrl_step,
                max_episode_length=self.max_episode_length,
                mode=mode,
                bpm=bpm,
                target_force=target_force_val
            )
        else:
            bpm_range = (
                getattr(self.cfg.rewards, "bpm_min", 60.0),
                getattr(self.cfg.rewards, "bpm_max", 160.0)
            )
            prob_rest = getattr(self.cfg.rewards, "prob_rest", 0.3)
            prob_double = getattr(self.cfg.rewards, "prob_double", 0.2)
            
            self.rhythm_generator = RhythmGenerator(
                num_envs=self.num_envs,
                device=self.device,
                dt=self.dt_ctrl_step,
                max_episode_length=self.max_episode_length,
                bpm_range=bpm_range,
                prob_rest=prob_rest,
                prob_double=prob_double,
                target_force=target_force_val
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

    # ----------------------------------------------------------------------
    # [New] 座標系変換ヘルパー (Sim:Down+ <-> Project:Up+)
    # ----------------------------------------------------------------------
    def _get_corrected_joint_state(self):
        """
        Sim(負方向=正) から取得した関節状態を、プロジェクト仕様(正方向=正)に変換して返す。
        対象: 手首(0) と グリップ(1) 両方の符号を反転。
        """
        # 全関節を取得
        q_full = self.robot.data.joint_pos
        qd_full = self.robot.data.joint_vel
        
        # 対象DOFを抽出 [Batch, 2]
        q = q_full[:, self.dof_idx].clone()
        qd = qd_full[:, self.dof_idx].clone()
        
        # ★修正: 全ての対象軸(0, 1)の符号を反転
        q *= -1.0
        qd *= -1.0
        
        return q, qd

    def _get_corrected_full_state(self):
        """コントローラやログ用に、全関節配列(q_full)の対象軸だけ反転したものを返す"""
        q_full = self.robot.data.joint_pos.clone()
        qd_full = self.robot.data.joint_vel.clone()
        
        # ★修正: 対象軸のインデックス全てで反転
        wrist_idx, grip_idx = self.joint_ids_tuple
        q_full[:, [wrist_idx, grip_idx]] *= -1.0
        qd_full[:, [wrist_idx, grip_idx]] *= -1.0
        
        return q_full, qd_full

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
        # -- (1) Pre-physics --
        self._pre_physics_step(actions)
        
        self.max_force_z_buffer[:] = 0.0
        
        # 安全策として履歴リストを用意
        force_history_list = []
        
        # P_cmd の値を保持する変数 (ログ用)
        # _apply_action で計算されるが、ここでログ用に取得したい場合は
        # _apply_action の戻り値にするか、コントローラから再取得する
        # ここでは後者(get_last_telemetry)を使う
        
        # -- (2) Physics Step (Decimation Loop) --
        for i in range(self.cfg.decimation):
            
            # 1. アクション適用 (トルク計算・適用のみ)
            self._apply_action()
            
            # 2. 物理シミュレーション進行
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(dt=self.cfg.sim.dt)
            
            # 3. センサーデータ取得
            if self.stick_sensor.data.net_forces_w.dim() == 3:
                current_force_vec = self.stick_sensor.data.net_forces_w[:, 0, :] 
            else:
                current_force_vec = self.stick_sensor.data.net_forces_w 
            
            force_history_list.append(current_force_vec.clone())
            
            current_force_z = current_force_vec[:, 2].clamp(min=0.0)
            self.max_force_z_buffer = torch.max(self.max_force_z_buffer, current_force_z)

            # 4. ★ログバッファリング (ここで一本化)
            if self.logging_manager.enable_logging:
                # 時刻を進める (dt = 5ms)
                self.logging_manager.update_time(self.cfg.sim.dt)
                
                # 情報取得
                current_steps = self.episode_length_buf
                tgt_val = self.rhythm_generator.get_current_target(current_steps)[0].item()
                tgt_bpm = self.rhythm_generator.current_bpms[0].item()
                
                # テレメトリ取得
                telemetry = self.action_controller.get_last_telemetry()
                if telemetry is None: telemetry = {}
                
                # Model Cの補完など (必要なら)
                if self.actuator_net is not None:
                     nan_tensor = torch.full((self.num_envs, 3), float('nan'), device=self.device)
                     if "P_cmd" not in telemetry: telemetry["P_cmd"] = nan_tensor
                     if "P_out" not in telemetry: telemetry["P_out"] = nan_tensor

                # ★修正: ログには補正後の値を渡す
                q_log, qd_log = self._get_corrected_full_state()
                
                # バッファに追加
                self.logging_manager.buffer_step_data(
                    q_full=q_log,   # <--- 修正
                    qd_full=qd_log, # <--- 修正
                    telemetry=telemetry,
                    actions=self.actions,
                    current_sim_time=self.sim.current_time,
                    target_force=tgt_val,
                    target_bpm=tgt_bpm
                )

        # -- (3) Post-processing (RL Step) --
        self.episode_length_buf += 1
        
        obs = self._get_observations()
        
        target_force_val = getattr(self.cfg.rewards, "target_force_fd", 20.0)
        target_force_tensor = torch.full((self.num_envs,), target_force_val, device=self.device)
        rew, reward_terms = self._get_rewards(force_max=self.max_force_z_buffer, target_ref=target_force_tensor)
        
        self.reset_time_outs = self.episode_length_buf >= self.max_episode_length
        self.reset_terminated[:] = False
        terminated, time_outs = self._get_dones()
        
        # Reset Mask Definition
        reset_mask = terminated | time_outs

        # 5. ★ログ書き込み確定
        if self.logging_manager.enable_logging:
            force_history_tensor = torch.stack(force_history_list, dim=1) # [N, Decimation, 3]
            
            f1_val = torch.zeros_like(rew)
            if hasattr(self.reward_manager, "get_first_hit_force"):
                 f1_val = self.reward_manager.get_first_hit_force()
            
            self.logging_manager.finalize_log_step(
                peak_force=force_history_tensor,
                f1_force=f1_val,
                step_reward=rew
            )

        # RSL RL Logging
        if not hasattr(self, "extras"): self.extras = {}
        self.extras["log"] = {}

        if not hasattr(self, "episode_sums"):
            self.episode_sums = {
                k: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for k in reward_terms.keys()
            }
            self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.episode_sums["total"] += rew
        for key, value in reward_terms.items():
            self.episode_sums[key] += value

        if reset_mask.any():
            env_ids = reset_mask.nonzero(as_tuple=False).flatten()
            episode_info = {}
            episode_info["reward"] = self.episode_sums["total"][env_ids]
            for key, count in self.episode_sums.items():
                if key != "total":
                    episode_info[key] = count[env_ids]

            self.extras["episode"] = episode_info
            self.episode_sums["total"][env_ids] = 0.0
            for key in self.episode_sums.keys():
                self.episode_sums[key][env_ids] = 0.0
            self._reset_idx(env_ids)
        else:
            if "episode" in self.extras:
                self.extras.pop("episode")
        
        for k, v in reward_terms.items():
             self.extras["log"][f"Step_Reward/{k}"] = torch.mean(v)

        self.extras["time_outs"] = time_outs
        self.extras["force/max_force_pooled"] = self.max_force_z_buffer.mean()

        current_len = self.episode_length_buf[0].item() if isinstance(self.episode_length_buf, torch.Tensor) else self.episode_length_buf
        should_save = (current_len % 50 == 0) or (reset_mask.any().item())

        if self.logging_manager.enable_logging and should_save:
             if self.logging_manager.logger is not None:
                 self.logging_manager.logger.save()

        return obs, rew, terminated, time_outs, self.extras

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions is not None:
            self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        """アクションの適用 (モデルの切り替えロジックを含む)"""
        
        # 1. 基本のアクション
        raw_cmd = (self.actions + 1.0) / 2.0        # -1~1を0~1に変換
        
        # P_cmd (3ch) の取得
        # TorqueActionController の compute_pressure が [Batch, 3] を返すと仮定
        # (返さない場合は TorqueActionController 側の実装確認が必要ですが、標準的な実装であれば返します)
        p_cmd_3d = None
        if hasattr(self.action_controller, "compute_pressure"):
            with torch.no_grad():
                p_cmd_3d = self.action_controller.compute_pressure(self.actions)
        
        # --- [修正] ActuatorNet (Model C) の適用ロジック ---
        if self.actuator_net is not None and isinstance(self.actuator_net, CascadedActuatorNet):
             # 必要な状態量を取得
             if p_cmd_3d is None:
                 # フォールバック: もしコントローラが返さない場合、簡易計算 (ここは実装に合わせて調整)
                 # 例: raw_cmd がそのまま圧力比率なら Pmax を掛けるなど
                 p_cmd_3d = torch.zeros((self.num_envs, 3), device=self.device)
            
             # ★修正: ヘルパーを使って「上=正」の状態を取得
             theta, theta_dot = self._get_corrected_joint_state()
             
             # 推論実行: (P_cmd, P_prev, Theta, Theta_dot) -> (Force, P_est)
             force, p_est = self.actuator_net(p_cmd_3d, self.last_pressure_est, theta, theta_dot)
             
             # 状態更新
             self.last_pressure_est = p_est.detach()

             # =========================================================
             # ★追加: Model C 使用時もログ用にテレメトリを更新する
             # =========================================================
             # ActuatorNetの推定内圧 (p_est) を P_out としてログに残す
             if self.action_controller is not None:
                 self.action_controller._last_telemetry = {
                     "P_cmd": p_cmd_3d.clone(),
                     "P_out": p_est.clone(),     # <--- これでログに反映されます
                     # 以下は必須ではないが形式を合わせるためのプレースホルダ
                     "tau_w": torch.zeros_like(p_est[:,0]), 
                     "tau_g": torch.zeros_like(p_est[:,0])
                 }
             
             # トルク計算: Tau = r * (Pull_F - Pull_DF)
             # Index: 0=DF, 1=F, 2=G (cfg/actuator_net_cfg.py の定義に準拠)
             r = self.action_controller.cfg.r
             
             # 手首: 屈曲(F) - 背屈(DF)  ※注: torque.py の修正と合わせるなら符号に注意
             # ここでは「DFが引くと正(上)」となる定義と仮定
             tau_wrist = r * (force[:, 0] - force[:, 1]) # DF - F
             tau_grip  = r * (- force[:, 2])
             
             torques = torch.stack([tau_wrist, tau_grip], dim=1)

            # ★重要: Simに書き込む直前に手首トルクを反転 (Simは上が負なので、正のトルクを伝えるには負にする)
             torques *= -1.0
             
             # ロボットへ適用 (Model Cの場合はここで終了)
             self.robot.set_joint_effort_target(torques, joint_ids=self.joint_ids_tuple)
             return 

        # --- 以下、既存の Model A/B ロジック (Model Cじゃない場合のみ実行) ---

        
        processed_actions = self.actions

        # コントローラに渡す座標系の補正 (Sim:Down+ -> Real:Up+)
        q_full_corrected, _ = self._get_corrected_full_state()
        
        self.action_controller.apply(
            actions=processed_actions,
            q=q_full_corrected,
            joint_ids=self.joint_ids_tuple,
            robot=self.robot
        )


    def _get_observations(self) -> dict:
        q, qd = self._get_corrected_joint_state()
        
        current_steps = self.episode_length_buf
        bpm_obs = self.rhythm_generator.current_bpms.view(self.num_envs, 1) / 180.0
        rhythm_buf = self.rhythm_generator.get_lookahead(current_steps, self.lookahead_steps)
        target_force = getattr(self.cfg.rewards, "target_force_fd", 20.0)
        rhythm_buf = rhythm_buf / target_force

        obs = torch.cat((q, qd, bpm_obs, rhythm_buf), dim=-1)
        return {"policy": obs}

    def _get_rewards(self, force_max: torch.Tensor = None, target_ref: torch.Tensor = None) -> torch.Tensor:
        if force_max is None: force_max = self.max_force_z_buffer
        
        dt_step = self.cfg.sim.dt * self.cfg.decimation
        
        current_steps = self.episode_length_buf
        target_trace = self.rhythm_generator.get_current_target(current_steps).view(-1)
        
        if target_ref is None:
            val = getattr(self.cfg.rewards, "target_force_fd", 20.0)
            target_ref = torch.full((self.num_envs,), val, device=self.device)

        # ★修正: joint_pos に補正済みの値を渡す
        q_corr, _ = self._get_corrected_joint_state()

        total_reward, reward_terms = self.reward_manager.compute_rewards(
            actions=self.actions,
            joint_pos=q_corr,
            force_z=force_max, 
            target_force_trace=target_trace, 
            target_force_ref=target_ref,
            dt=dt_step
        )
        
        if not hasattr(self, "extras"): self.extras = {}

        return total_reward, reward_terms

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.reset_terminated, self.reset_time_outs

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        if hasattr(self, "event_manager") and self.event_manager is not None:
            self.event_manager.reset(env_ids)

        # 1. 現在の関節位置バッファを複製
        q_target = self.robot.data.joint_pos[env_ids].clone()
        qd_target = self.robot.data.joint_vel[env_ids].clone()
        
        # 1. assets.py から読み込んだ定数 (ラジアン)
        val_wrist = torch.tensor(WRIST_J0, device=self.device)
        val_grip  = torch.tensor(GRIP_J0,  device=self.device)
        
        # 2. インデックスを使って書き換え
        wrist_idx = self.dof_idx[0]
        grip_idx  = self.dof_idx[1]
        
        # ★修正: Simへの書き込みなので、符号を反転させる (Code(Up+) -> Sim(Down+))
        q_target[:, wrist_idx] = -val_wrist # <--- 反転
        q_target[:, grip_idx]  = -val_grip
        qd_target[:] = 0.0
        
        # 3. 物理エンジンへの書き込み
        self.robot.write_joint_state_to_sim(
            position=q_target, 
            velocity=qd_target, 
            env_ids=env_ids
        )

        if hasattr(self, "reward_manager"):
            self.reward_manager.reset_idx(env_ids)
        if hasattr(self, "logging_manager"):
            self.logging_manager.reset_idx(env_ids)
        if hasattr(self, "rhythm_generator"):
            self.rhythm_generator.reset(env_ids)
            

        # --- [追加] Model C用のリセット ---
        if hasattr(self, "last_pressure_est"):
            self.last_pressure_est[env_ids] = 0.0

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
