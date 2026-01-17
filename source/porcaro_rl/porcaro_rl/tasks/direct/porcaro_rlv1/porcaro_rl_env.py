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
from .simple_rhythm_generator import SimpleRhythmGenerator
from .cfg.assets import WRIST_J0, GRIP_J0


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

        # =========================================================
        # ★ [修正 1] サブステップ間の最大力を保持するバッファの追加
        # =========================================================
        # 物理ステップ間(5ms)に発生したスパイクを逃さないためのMax Pooling用バッファ
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

        # =========================================================
        # ★ 変更箇所: リズム生成器の切り替えロジック
        # =========================================================
        self.dt_ctrl_step = self.cfg.sim.dt * self.cfg.decimation
        
        # Configに追加したフラグを確認 (デフォルトFalseとして安全に取得)
        use_simple = getattr(self.cfg, "use_simple_rhythm", False)

        target_force_val = getattr(self.cfg.rewards, "target_force_fd", 20.0)

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
                target_force=target_force_val # ★引数追加
            )
        else:
            # 本番学習用のランダム生成器を使用
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
        """
        DirectRLEnvのstepをオーバーライドし、物理サブステップごとのセンサ更新と
        力の最大値取得(Max Pooling)、およびロギング処理を行う。
        """
        # 1. アクションの前処理
        self._pre_physics_step(actions)

        # Max Pooling用バッファのリセット
        self.max_force_z_buffer[:] = 0.0
        
        # ログ用の力履歴バッファ (Decimation中の全データを記録するため)
        force_history_list = []
        
        # ターゲット力の取得
        target_force_val = getattr(self.cfg.rewards, "target_force_fd", 20.0)
        target_force_tensor = torch.full((self.num_envs,), target_force_val, device=self.device)

        # ---------------------------------------------------------
        # 物理サブステップループ (200Hz)
        # ---------------------------------------------------------
        for _ in range(self.cfg.decimation):
            # (a) アクション適用 & ログ収集 (buffer_step_dataはここで呼ばれる)
            self._apply_action()
            
            # (b) 物理計算
            self.scene.write_data_to_sim()
            self.sim.step()
            
            # (c) センサ更新 (必須)
            self.scene.update(dt=self.cfg.sim.dt)

            # (d) データの取得
            # フィルタありの場合 (num_envs, num_filters, 3) なので [:, 0, :] を指定
            if self.stick_sensor.data.net_forces_w.dim() == 3:
                current_force_vec = self.stick_sensor.data.net_forces_w[:, 0, :] # (N, 3)
            else:
                current_force_vec = self.stick_sensor.data.net_forces_w # (N, 3)

            # ログ用に履歴を保存 (.clone() が重要)
            force_history_list.append(current_force_vec.clone())

            # Max Pooling (Z軸)
            current_force_z = current_force_vec[:, 2].clamp(min=0.0)
            self.max_force_z_buffer = torch.max(self.max_force_z_buffer, current_force_z)

        # ---------------------------------------------------------
        # ポスト処理 (50Hz)
        # ---------------------------------------------------------
        self.episode_length_buf += 1
        
        obs = self._get_observations()
        
        # 報酬計算
        rew = self._get_rewards(force_max=self.max_force_z_buffer, target_ref=target_force_tensor)
        
        terminated, time_outs = self._get_dones()
        
        # 環境リセット
        reset_mask = terminated | time_outs
        if reset_mask.any():
            reset_env_ids = reset_mask.nonzero(as_tuple=False).flatten()
            self._reset_idx(reset_env_ids)
        
        # =========================================================
        # ★ [修正箇所] ログのファイナライズ (ここが抜けていました)
        # =========================================================
        if self.logging_manager.enable_logging:
            # 履歴をテンソル化: (N, T, 3)
            force_history_tensor = torch.stack(force_history_list, dim=1)
            
            # Isaac Labの慣習に合わせて「最新が先頭」になるように時間を反転させる
            # (LoggingManager内で flip される場合、ここで chrono順 だと逆になるため調整)
            # もし LoggingManager が flip するなら、ここでは Chrono順 (stackしたまま) で渡すと
            # Reverse-Chrono (Newest First) に変換される。
            # 通常、センサー履歴は Newest First なので、それに合わせるため反転しておくのが無難。
            force_history_tensor = torch.flip(force_history_tensor, dims=[1])
            
            self.logging_manager.finalize_log_step(
                peak_force=force_history_tensor,
                f1_force=torch.zeros_like(rew), # 現状F1スコア計算がなければダミー
                step_reward=rew
            )
            
        # Extrasの作成
        extras = {
            "time_outs": time_outs,
            "force/max_force_pooled": self.max_force_z_buffer.mean()
        }
        
        if hasattr(self, "extras"):
            extras.update(self.extras)

        return obs, rew, terminated, time_outs, extras

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions is not None:
            self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        """アクションの適用 (モデルの切り替えロジックを含む)"""
        
        # 1. 基本のアクション (Neural Networkの出力: -1 ~ 1)
        raw_cmd = (self.actions + 1.0) / 2.0
        
        # 変更箇所: P_cmd (ダイナミクス適用前の指令圧力) を計算して保持
        # NNの出力(self.actions)が意図している圧力を計算します
        p_cmd_log = None
        if hasattr(self.action_controller, "compute_pressure"):
            with torch.no_grad():
                p_cmd_log = self.action_controller.compute_pressure(self.actions)
        
        # 2. ActuatorNet (Model C)
        if self.actuator_net is not None:
             # ※ ActuatorNetの場合、アクション自体がトルクや別次元になる可能性があるため
             # 従来のP_cmd/P_outはNaNにすべきかもしれません。
             # ここでは既存ロジックを維持しつつ、後続の処理に委ねます
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
        
        # この apply 内で P_out (processed_actionsに基づく圧力) と tau が計算・保存されます
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
            
            # 変更箇所: コントローラから取得したテレメトリに P_cmd を注入
            telemetry = self.action_controller.get_last_telemetry()
            if telemetry is None:
                telemetry = {}
            
            # P_cmd を辞書に追加
            if p_cmd_log is not None:
                telemetry["P_cmd"] = p_cmd_log
            elif self.actuator_net is not None:
                # Model Cで圧力が定義できない場合はNaN埋め
                nan_tensor = torch.full((self.num_envs, 3), float('nan'), device=self.device)
                telemetry["P_cmd"] = nan_tensor
                telemetry["P_out"] = nan_tensor # Controllerが出すP_outも無効化
            
            self.logging_manager.buffer_step_data(
                q_full=self.robot.data.joint_pos,
                qd_full=self.robot.data.joint_vel,
                telemetry=telemetry,
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

    def _get_rewards(self, force_max: torch.Tensor = None, target_ref: torch.Tensor = None) -> torch.Tensor:
        # 引数が無い場合のフォールバック (基本はstepから渡される)
        if force_max is None: force_max = self.max_force_z_buffer
        
        dt_step = self.cfg.sim.dt * self.cfg.decimation
        
        # 波形データ (休符判定などに使用)
        current_steps = self.episode_length_buf
        target_trace = self.rhythm_generator.get_current_target(current_steps).view(-1)
        
        # もしstepから渡されていなければConfigから作る
        if target_ref is None:
            val = getattr(self.cfg.rewards, "target_force_fd", 20.0)
            target_ref = torch.full((self.num_envs,), val, device=self.device)

        # Managerへ委譲
        total_reward, reward_terms = self.reward_manager.compute_rewards(
            actions=self.actions,
            joint_pos=self.robot.data.joint_pos[:, self.dof_idx],
            force_z=force_max, 
            target_force_trace=target_trace, # 休符判定用
            target_force_ref=target_ref,     # ★打撃評価用 (新規追加引数)
            dt=dt_step
        )
        
        # ログ用Extras
        if not hasattr(self, "extras"): self.extras = {}
        for key, val in reward_terms.items():
            self.extras[f"Reward/{key}"] = torch.mean(val)
            
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.reset_terminated, self.reset_time_outs

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        if hasattr(self, "event_manager") and self.event_manager is not None:
            self.event_manager.reset(env_ids)

        # =========================================================
        # ★ 修正箇所: 正しいAPIを使用した初期姿勢の強制オーバーライド ★
        # =========================================================
        
        # 1. 現在の関節位置バッファを複製（ベースとして使用）
        #    (num_envs, num_dof) の形状
        q_target = self.robot.data.joint_pos[env_ids].clone()
        qd_target = self.robot.data.joint_vel[env_ids].clone()
        
        # 1. assets.py から読み込んだ定数 (すでにラジアン) を Tensor化
        val_wrist = torch.tensor(WRIST_J0, device=self.device)
        val_grip  = torch.tensor(GRIP_J0,  device=self.device)
        
        # 2. インデックスを使って書き換え
        wrist_idx = self.dof_idx[0]
        grip_idx  = self.dof_idx[1]
        
        q_target[:, wrist_idx] = val_wrist
        q_target[:, grip_idx]  = val_grip
        
        qd_target[:] = 0.0
        
        # 3. 物理エンジンへの書き込み
        self.robot.write_joint_state_to_sim(
            position=q_target, 
            velocity=qd_target, 
            env_ids=env_ids
        )
        # =========================================================

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