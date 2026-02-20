# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/porcaro_rl_env.py
from __future__ import annotations
import argparse
import torch
import math
from collections.abc import Sequence
from collections import deque

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

        # [追加]: カリキュラム学習用のステップカウンタ
        self.total_env_steps = 0
        
        # [追加]: カリキュラム閾値 (累積ステップ数)総ステップ数737280000
        # Lv0 -> Lv1: 100 iters (約70M steps)
        # Lv1 -> Lv2: +200 iters (累積 約210M steps)
        self.curriculum_thresholds = [200_000_000, 400_000_000]

        # 物理パラメータを変えずに、強化学習が見る値だけを実機スケールに合わせる
        self.force_scale_sim_to_real = 1.0

        # [追加]: BPMごとの報酬ログ用バッファ (移動平均用)
        # キー: BPM(int), 値: deque(maxlen=20) ← 直近20回分のエピソード平均を保持
        self.bpm_reward_history = {
            bpm: deque(maxlen=20) 
            for bpm in [60, 80, 100, 120, 140, 160]
        }

        # ★ [追加]: パターンごとの報酬ログ用バッファ
        self.pattern_keys = ["single_4", "single_8", "double", "rest"]
        self.pattern_reward_history = {
            pat: deque(maxlen=20) for pat in self.pattern_keys
        }

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

        self.prev_actions = torch.zeros_like(self.actions)

        # =========================================================
        # ★ [維持] サブステップ間の最大力を保持するバッファ
        # =========================================================
        self.max_force_z_buffer = torch.zeros(self.num_envs, device=self.device)
        # [追加]: 平均力計算用の積算バッファ
        self.force_sum_buffer = torch.zeros(self.num_envs, device=self.device)

        # [追加]: 動的エピソード長管理用 (BPMに応じて終了ステップが変わる)
        self.episode_duration_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # ---------------------------------------------------------
        # 2. アクションコントローラの初期化
        # ---------------------------------------------------------
        dt_ctrl = self.cfg.sim.dt
        ctrl_cfg = self.cfg.controller

        pam_tau_scale_range = getattr(self.cfg, "pam_tau_scale_range", (1.0, 1.0))
        
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
            pam_tau_scale_range=pam_tau_scale_range,
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
        self.target_hit_force = getattr(self.cfg, "target_hit_force", 20.0)

        self.rhythm_generator = RhythmGenerator(
            num_envs=self.num_envs,
            device=self.device,
            dt=self.dt_ctrl_step,
            max_episode_length=self.max_episode_length, # Configで確保した最大長(20s分)
            bpm_range=self.cfg.bpm_range,
            target_force=self.target_hit_force
        )

        # Config設定に基づきモード（学習用/検証用）を切り替え
        use_simple = getattr(self.cfg, "use_simple_rhythm", False)
        if use_simple:
            mode = getattr(self.cfg, "simple_rhythm_mode", "double")
            bpm = getattr(self.cfg, "simple_rhythm_bpm", 160.0)
            print(f"[INFO] RhythmGenerator: Test Mode Enabled (Pattern: {mode}, BPM: {bpm})")
            self.rhythm_generator.set_test_mode(enabled=True, bpm=bpm, pattern=mode)
        else:
            print(f"[INFO] RhythmGenerator: Training Mode (Random BPM & Rudiments)")
            self.rhythm_generator.set_test_mode(enabled=False)
        
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
        # =========================================================
        # [追加]: カリキュラムレベルの更新ロジック
        # =========================================================
        # 今回のステップ数を加算 (並列数分だけ経験が進む)
        self.total_env_steps += self.num_envs
        
        # 現在のレベルを判定
        current_level = 0
        if self.total_env_steps > self.curriculum_thresholds[1]:
            current_level = 2
        elif self.total_env_steps > self.curriculum_thresholds[0]:
            current_level = 1
            
        # Generatorに適用 (変更がある場合のみログ出力などをしても良い)
        if hasattr(self, "rhythm_generator"):
            # レベルが上がった瞬間だけ通知したい場合はここで判定を入れる
            if self.rhythm_generator.curriculum_level != current_level:
                print(f"[Curriculum] Level Up! {self.rhythm_generator.curriculum_level} -> {current_level} (Steps: {self.total_env_steps})")
                self.rhythm_generator.set_curriculum_level(current_level)
        
        # -- (1) Pre-physics --
        self._pre_physics_step(actions)
        
        self.max_force_z_buffer[:] = 0.0
        
        # GPU上でメモリを一度だけ確保し、インデックスで書き込む方が圧倒的に高速です
        # shape: [NumEnvs, Decimation, 3]
        force_history_tensor = torch.zeros(
            (self.num_envs, self.cfg.decimation, 3), 
            device=self.device, 
            dtype=torch.float32
        )
        
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
            # clone() は最小限に。必要なデータだけをバッファに入れる
            if self.stick_sensor.data.net_forces_w.dim() == 3:
                current_force_vec = self.stick_sensor.data.net_forces_w[:, 0, :] 
            else:
                current_force_vec = self.stick_sensor.data.net_forces_w

            current_force_vec = current_force_vec * self.force_scale_sim_to_real
            
            # ★改善点2: Tensorへの直接代入 (高速化)
            force_history_tensor[:, i, :] = current_force_vec
            
            # 以前: current_force_z = current_force_vec[:, 2].clamp(min=0.0)
            current_force_mag = torch.norm(current_force_vec, dim=-1)
            
            self.max_force_z_buffer = torch.max(self.max_force_z_buffer, current_force_mag)
            
            # [追加]: 平均計算用に積算
            self.force_sum_buffer += current_force_mag

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
                q_log, qd_log = self._get_corrected_full_state()
                
                # Model Cの補完など (必要なら)
                if self.actuator_net is not None:
                     nan_tensor = torch.full((self.num_envs, 3), float('nan'), device=self.device)
                     if "P_cmd" not in telemetry: telemetry["P_cmd"] = nan_tensor
                     if "P_out" not in telemetry: telemetry["P_out"] = nan_tensor

                # ★修正: ログには補正後の値を渡す
                q_log, qd_log = self._get_corrected_full_state()
                
                # バッファに追加
                self.logging_manager.buffer_step_data(
                    q_full=q_log,
                    qd_full=qd_log,
                    telemetry=telemetry,
                    actions=self.actions,
                    current_sim_time=self.sim.current_time,
                    target_force=self.rhythm_generator.get_current_target(self.episode_length_buf)[0].item(),
                    target_bpm=self.rhythm_generator.current_bpms[0].item()
                )

        # -- (3) Post-processing (RL Step) --
        self.episode_length_buf += 1
        
        obs = self._get_observations()

        # =========================================================
        # ★ [重要] Observation Sanitization (NaNガード)
        # =========================================================
        # 観測値にNaNやInfが含まれていたら、0.0に置換して「なかったこと」にする
        # これをやらないと、一瞬の爆発で学習全体がクラッシュします
        # obs は {"policy": Tensor} の辞書形式なので、中身を走査してガードします
        if isinstance(obs, dict):
            for key, val in obs.items():
                if isinstance(val, torch.Tensor):
                    if torch.isnan(val).any() or torch.isinf(val).any():
                        # NaN/Infがあったら0.0に置換して書き戻す
                        obs[key] = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
        elif isinstance(obs, torch.Tensor):
            # 万が一 Tensor が直接返ってきた場合のフォールバック
            if torch.isnan(obs).any() or torch.isinf(obs).any():
                obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 報酬のガード (こちらはTensorなのでそのままでOK)
        target_force_tensor = torch.full((self.num_envs,), self.target_hit_force, device=self.device)
        rew, reward_terms = self._get_rewards(force_max=self.max_force_z_buffer, target_ref=target_force_tensor)
        
        if torch.isnan(rew).any() or torch.isinf(rew).any():
            rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.reset_terminated[:] = False
        terminated, time_outs = self._get_dones()
        
        # Reset Mask Definition
        reset_mask = terminated | time_outs
        

        # 5. ログ書き込み確定
        if self.logging_manager.enable_logging:
            # force_history_tensor は既に Tensor なので stack 不要
            f1_val = torch.zeros_like(rew)
            if hasattr(self.reward_manager, "get_first_hit_force"):
                 f1_val = self.reward_manager.get_first_hit_force()
            
            self.logging_manager.finalize_log_step(
                peak_force=force_history_tensor, # そのまま渡す
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
            # --- [修正箇所 START] ---
            if hasattr(self, "rhythm_generator"):
                done_bpms = self.rhythm_generator.current_bpms[env_ids]
                done_pat_idxs = self.rhythm_generator.current_pattern_idxs[env_ids]
                done_rewards = self.episode_sums["total"][env_ids]
                

                # 1. BPMごとの集計
                for t_bpm in [60.0, 80.0, 100.0, 120.0, 140.0, 160.0]:
                    bpm_mask = (torch.abs(done_bpms - t_bpm) < 1.0)
                    if bpm_mask.any():
                        avg_bpm_reward = done_rewards[bpm_mask].mean().item()
                        self.bpm_reward_history[int(t_bpm)].append(avg_bpm_reward)

                # 2. パターンごとの集計
                for pat_idx, pat_name in enumerate(self.pattern_keys):
                    pat_mask = (done_pat_idxs == pat_idx)
                    if pat_mask.any():
                        avg_pat_reward = done_rewards[pat_mask].mean().item()
                        self.pattern_reward_history[pat_name].append(avg_pat_reward)
            # --- [修正箇所 END] ---
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

        # ------------------------------------------------------------------
        # ★ [追加] ここで毎ステップ、バッファにある最新の平均値をログに載せる
        # ------------------------------------------------------------------
        for t_bpm, history in self.bpm_reward_history.items():
            if len(history) > 0:
                self.extras["log"][f"Episode_Reward/BPM_{t_bpm}"] = sum(history) / len(history)
            else:
                self.extras["log"][f"Episode_Reward/BPM_{t_bpm}"] = 0.0

        # 2. パターン別ログ
        for pat_name, history in self.pattern_reward_history.items():
            if len(history) > 0:
                self.extras["log"][f"Episode_Reward/Pattern_{pat_name}"] = sum(history) / len(history)
            else:
                self.extras["log"][f"Episode_Reward/Pattern_{pat_name}"] = 0.0

        # Step Rewardのログなど (既存)
        for k, v in reward_terms.items():
             self.extras["log"][f"Step_Reward/{k}"] = torch.mean(v)

        self.extras["time_outs"] = time_outs
        self.extras["force/max_force_pooled"] = self.max_force_z_buffer.mean()

        current_len = self.episode_length_buf[0].item() if isinstance(self.episode_length_buf, torch.Tensor) else self.episode_length_buf
        should_save = (current_len % 50 == 0) or (reset_mask.any().item())

        if self.logging_manager.enable_logging and should_save:
             if self.logging_manager.logger is not None:
                 self.logging_manager.logger.save()

        self.prev_actions[:] = self.actions

        return obs, rew, terminated, time_outs, self.extras

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions is not None:
            self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        """アクションの適用 (モデルの切り替えロジックを含む)"""
        
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
        
        # [変更]: BPM正規化
        bpm_val = self.rhythm_generator.current_bpms.view(-1, 1)
        bpm_obs = bpm_val / 180.0

        # [追加]: 位相信号 (Phase Signal)
        # 1拍 = 0~2pi となる位相
        time_s = self.episode_length_buf.float().unsqueeze(1) * self.dt_ctrl_step
        phase = time_s * (bpm_val / 60.0) * (2 * math.pi)
        sin_phase = torch.sin(phase)
        cos_phase = torch.cos(phase)
        
        # 先読み情報
        current_steps = self.episode_length_buf
        rhythm_buf = self.rhythm_generator.get_lookahead(current_steps, self.lookahead_steps)
        rhythm_buf = rhythm_buf / self.target_hit_force

        # [変更]: 観測結合 (q, qd, prev_act, sin, cos, bpm, lookahead)
        # 観測次元: q(2)+qd(2)+prev_act(3)+sin(1)+cos(1)+bpm(1)+lookahead(25) = 35次元
        obs = torch.cat((q, qd, self.prev_actions, sin_phase, cos_phase, bpm_obs, rhythm_buf), dim=-1)
        return {"policy": obs}

    def _get_rewards(self, force_max: torch.Tensor = None, target_ref: torch.Tensor = None) -> torch.Tensor:
        if force_max is None: force_max = self.max_force_z_buffer
        dt_step = self.cfg.sim.dt * self.cfg.decimation
        current_steps = self.episode_length_buf
        target_trace = self.rhythm_generator.get_current_target(current_steps).view(-1)
        if target_ref is None:
            target_ref = torch.full((self.num_envs,), self.target_hit_force, device=self.device)
        q_corr, _ = self._get_corrected_joint_state()

        # [追加] BPM情報をRewardManagerに渡す
        current_bpms = self.rhythm_generator.current_bpms if hasattr(self, "rhythm_generator") else None

        total_reward, reward_terms = self.reward_manager.compute_rewards(
            actions=self.actions,
            joint_pos=q_corr,
            force_z=force_max, 
            target_force_trace=target_trace, 
            target_force_ref=target_ref,
            dt=dt_step,
            current_bpm=current_bpms  # <--- ここで渡す
        )
        if not hasattr(self, "extras"): self.extras = {}
        return total_reward, reward_terms

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # [変更]: 動的タイムアウト判定
        # BPMに基づく「4小節終了ステップ数」を超えたらタイムアウト
        time_outs = self.episode_length_buf >= self.episode_duration_steps
        
        # 安全策: バッファ溢れ防止
        time_outs |= (self.episode_length_buf >= self.max_episode_length - 1)

        # 2. ★追加：数値発散（爆発）の検知
        # 関節角度が異常な値（例：3.14rad以上やNaN）になったら強制終了
        q, _ = self._get_corrected_joint_state()
        is_exploded = torch.any(torch.abs(q) > 4.0, dim=-1) # 角度が約230度を超えたら異常
        is_exploded |= torch.any(torch.isnan(q), dim=-1)    # NaNが出たら異常

        self.reset_terminated[:] = False
        return self.reset_terminated, time_outs

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        # ==========================================================
        # ★ 変更箇所: 初期角度のオーバーライド問題を解消するリセットロジック
        # 理由: 
        # 1. assets.py で設定された正しい初期値 (default_joint_pos) を直接使用する。
        # 2. 二重にマイナスを掛けてしまう手動上書きロジックを排除。
        # 3. EventManager を後に呼ぶことで、DR(Domain Randomization) のランダム化効果を潰さずに適用する。
        # ==========================================================
        
        # 1. assets.py の InitialStateCfg で設定されたデフォルトの関節位置/速度を取得
        q_target = self.robot.data.default_joint_pos[env_ids].clone()
        qd_target = self.robot.data.default_joint_vel[env_ids].clone()
        
        # 2. 物理エンジンへの初期値書き込み
        self.robot.write_joint_state_to_sim(
            position=q_target, 
            velocity=qd_target, 
            env_ids=env_ids
        )

        # 3. イベントマネージャによるリセット (DRなど) を後から適用
        # これにより、上記で書き込んだ default_joint_pos を基準としたスケールランダム化が正しく乗る
        if hasattr(self, "event_manager") and self.event_manager is not None:
            self.event_manager.reset(env_ids)

        # ==========================================================

        # [追加]: リズムリセット & エピソード長再計算
        if hasattr(self, "rhythm_generator"):
            self.rhythm_generator.reset(env_ids)
            
            # 4小節 = 16拍分の時間 [s]
            bpms = self.rhythm_generator.current_bpms[env_ids]
            durations_s = 16.0 * (60.0 / bpms)
            
            # ステップ数に換算
            steps = (durations_s / self.dt_ctrl_step).long()
            self.episode_duration_steps[env_ids] = steps
            
        # [追加]: 前回アクションのリセット
        self.prev_actions[env_ids] = 0.0

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
