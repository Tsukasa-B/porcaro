# porcaro_env.py
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
from isaaclab.sensors import ContactSensor, ContactSensorData # ドラムとセンサのために追加

# porcaro_rl_env.py のインポートセクション

# パッケージ実行時
from .porcaro_rl_env_cfg import PorcaroRLEnvCfg
from .actions.base import ActionController # IF (型ヒント用)
from .actions.torque import TorqueActionController
from .logging.logging_manager import LoggingManager # <-- 新規追加
from .rewards.reward import RewardManager          # <-- 新規追加




class PorcaroRLEnv(DirectRLEnv):
    """Porcaro 環境クラス (チュートリアルベース)"""

    # Cfg の型ヒントを PorcaroEnvCfg に変更
    cfg: PorcaroRLEnvCfg

    def __init__(self, cfg: PorcaroRLEnvCfg, render_mode: str | None = None, **kwargs):
        
        # アセット/センサ用の変数を初期化 (porcaro_rl_env.py を参考)
        self.robot: Articulation = None
        self.drum: RigidObject = None
        self.stick_sensor: ContactSensor = None
        self.drum_sensor: ContactSensor = None

        # --- 追加: アクションコントローラ ---
        self.action_controller: ActionController = None
        # --- 変更: ロガーと報酬マネージャの変数を初期化 ---
        self.logging_manager: LoggingManager = None
        self.reward_manager: RewardManager = None
        
        # 親クラスの __init__ を呼ぶ
        super().__init__(cfg, render_mode, **kwargs)

        # cfg.dof_names に基づいて、制御対象の関節インデックスを取得
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        
# --- 追加: dof_idx をタプル (wid, gid) に変換 ---
        # TorqueActionController が (wid, gid) を要求するため
        if len(self.dof_idx) != 2:
            raise ValueError(f"Expected 2 DOFs (wrist, grip), but found {len(self.dof_idx)} based on {self.cfg.dof_names}")
        self.joint_ids_tuple: tuple[int, int] = (self.dof_idx[0], self.dof_idx[1])
        
        # アクションバッファ
        # (cfg.action_space は 3 になっているはず)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # --- 追加: コントローラの初期化 ---
        # dt_ctrl = sim.dt * decimation
        dt_ctrl = self.cfg.sim.dt * self.cfg.decimation
        ctrl_cfg = self.cfg.controller
        
        self.action_controller = TorqueActionController(
            dt_ctrl=dt_ctrl,
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
        
        # --- 追加: コントローラの状態をリセット ---
        # (reset() は device と n_envs を要求するため __init__ の最後で呼ぶ)
        self.action_controller.reset(self.num_envs, self.device)

        # 1. ロギングマネージャ
        self.logging_manager = LoggingManager(
            cfg=self.cfg.logging,
            reward_cfg=self.cfg.reward_logging,
            dof_idx=self.dof_idx,
            num_envs=self.num_envs,
            device=self.device,
            dt=self.cfg.sim.dt,                  # <-- (新規追加) 物理ステップdt
            decimation=self.cfg.decimation      # <-- (新規追加) デシメーション
        )

        print(f"[DEBUG] ロギングを強制的に有効化します (Config上: {self.cfg.logging.enabled})")
        self.logging_manager.force_enable()
        # (追加) 報酬ログも強制有効化
        print(f"[DEBUG] 報酬ログを強制的に有効化します (Config上: {self.cfg.reward_logging.enabled})")
        self.logging_manager.force_enable_reward_log()
        
        # 2. 報酬マネージャ
        self.reward_manager = RewardManager(
            cfg=self.cfg.rewards,
            num_envs=self.num_envs,
            device=self.device
        )

    def _setup_scene(self):
        """シーンにアセット（ロボット、ドラム、地面、ライト、センサ）をセットアップする"""
        
        # ロボット (Cfg に基づきインスタンス化)
        self.robot = Articulation(self.cfg.robot_cfg)
        
        # ドラム (Cfg に基づきインスタンス化)
        self.drum = RigidObject(self.cfg.drum_cfg)

        # 地面
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # 環境の複製
        self.scene.clone_environments(copy_from_source=False)
        
        # シーンへのアセット登録
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["drum"] = self.drum # ドラムを登録
        
        # ライト
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # センサ (Cfg に基づきインスタンス化)
        self.stick_sensor = ContactSensor(self.cfg.stick_contact_cfg)
        self.drum_sensor = ContactSensor(self.cfg.drum_contact_cfg)
        
        # シーンへのセンサ登録
        self.scene.sensors["stick_contact"] = self.stick_sensor
        self.scene.sensors["drum_contact"] = self.drum_sensor
        
        # (チュートリアルにあったマーカー関連のコードは削除)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """シミュレーションステップの前にアクションをバッファする"""
        if actions is None:
            return
        
        # アクションを [-1, 1] にクリップして保存
        # (TorqueActionController が内部で [0, Pmax] にスケーリングする)
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        """バッファされたアクション（空気圧）をコントローラ経由でトルクとして適用する"""
        
        # --- 変更: TorqueActionController を使用 ---
        
        # 1. 現在の全関節位置 q を取得 (num_envs, num_total_dof)
        q_full = self.robot.data.joint_pos 
        
        # 2. TorqueActionController の apply を呼ぶ
        #    (コントローラが内部で robot.set_joint_effort_target を呼びます)
        self.action_controller.apply(
            actions=self.actions,
            q=q_full,
            joint_ids=self.joint_ids_tuple,
            robot=self.robot
        )

        # porcaro_rl_env.py の _apply_action メソッドの *末尾* に追加

        # (既存の robot.set_joint_effort_target(tau_full) などの後)

        # --- 変更: データロギング (マネージャに委譲) ---
        
        # 内部時刻を更新
        dt_ctrl = self.cfg.sim.dt * self.cfg.decimation
        self.logging_manager.update_time(dt_ctrl)
        
        # (変更) f1 を渡さず、buffer_step_data を呼ぶ
        self.logging_manager.buffer_step_data(
            q_full=q_full,
            qd_full=self.robot.data.joint_vel,
            telemetry=self.action_controller.get_last_telemetry()
        )

    def _get_observations(self) -> dict:
        """環境の観測を取得する"""
        
        # 全関節の「位置」を取得 (num_envs, num_total_dof)
        q = self.robot.data.joint_pos
        # 制御対象の関節の「位置」を抽出 (num_envs, 2)
        joint_pos_obs = q[:, self.dof_idx]
        
        # 全関節の「速度」を取得 (num_envs, num_total_dof)
        qd = self.robot.data.joint_vel
        # 制御対象の関節の「速度」を抽出 (num_envs, 2)
        joint_vel_obs = qd[:, self.dof_idx]
        
        # --- 変更: リズム情報 4次元をゼロ埋め (Placeholder) ---
        # 内訳: [time_to_next_hit, target_force, phase_signal, previous_reward]
        rhythm_info = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=torch.float32
        )
        
        # 観測 = [位置(2), 速度(2), リズム情報(4)] の合計 8 次元
        # (cfg.observation_space = 8 と一致させる)
        obs = torch.hstack((joint_pos_obs, joint_vel_obs, rhythm_info)) # <-- 変更
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """(変更) 報酬を計算し、ログを「書き込み確定」する"""

        # (A) センサーデータを取得
        force_history_tensor = self.stick_sensor.data.net_forces_w_history  # (num_envs, 32, 3)
        
        # (B) このRLステップ (decimation=4回) の間の履歴を抽出
        #    (num_envs, decimation, 3)
        current_rl_step_force_history = force_history_tensor[:, 0:self.cfg.decimation, :]
        
        # (C) このRLステップ間の「Z軸力」の履歴 (num_envs, decimation)
        force_z_history_in_step = current_rl_step_force_history[..., 2]
        
        
        # 1. 報酬と終了判定を計算 (f1 が内部で更新される)
        # (変更) RewardManager に (C) のZ軸力「履歴」を渡す
        rewards = self.reward_manager.compute_reward_and_dones(
            force_z_history=force_z_history_in_step,     # (C) Z軸の力履歴 (num_envs, decimation)
            episode_length_buf=self.episode_length_buf,    
            max_episode_length=(self.max_episode_length - 1) 
        )
        
        # 2. (変更なし) 更新された f1 と reward を取得
        f1_force_tensor = self.reward_manager.get_first_hit_force()
        
        # 3. (変更) バッファデータと最新データを組み合わせてログ書き込み
        self.logging_manager.finalize_log_step(
            peak_force=force_history_tensor, # (A) 物理ステップのピーク力「履歴」バッファ
            f1_force=f1_force_tensor,     # (RewardManagerが計算したF1)
            step_reward=rewards           # (RewardManagerが計算した報酬)
        )
        
        # 4. (変更なし) 終了フラグを親クラスに設定
        self.reset_terminated, self.reset_time_outs = self.reward_manager.get_dones()
        
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """報酬計算ステップで計算済みの終了判定を返す"""
        
        # --- 変更: RewardManager が計算した値を返す ---
        return self.reset_terminated, self.reset_time_outs

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """
        指定された環境IDの内部状態をリセットします。
        (DirectRLEnv のメソッドをオーバーライド)
        """
        
        # 1. 親クラスの標準リセット処理（ロボットの位置など）を実行
        super()._reset_idx(env_ids)
        
        # 2. (重要) RewardManager の内部状態をリセット
        # (これにより f1_force, hit_count などが 0 に戻る)
        if hasattr(self, "reward_manager"):
            self.reward_manager.reset_idx(env_ids)
            
        # 3. (重要) LoggingManager の内部状態（時刻）をリセット
        if hasattr(self, "logging_manager"):
            self.logging_manager.reset_idx(env_ids)

    def close(self):
        """
        環境が閉じられるときに呼ばれ、ログを保存する。
        (BaseEnv のメソッドをオーバーライド)
        """
        
        # 1. (重要) ログマネージャにログの保存を指示
        if hasattr(self, "logging_manager") and self.logging_manager is not None:
            print("[PorcaroRLEnv] close() が呼ばれました。ログを保存します...")
            self.logging_manager.save_on_exit()
        
        # 2. 親クラスの close() を呼ぶ (必須)
        super().close()


# --- 実行用 main 関数 (porcaro_rl_env.py から移植・調整) ---

def _parse_args():
    """コマンドライン引数をパースする"""
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true", help="UI なしで実行")
    p.add_argument("--num_envs", type=int, default=128, help="並列環境数") # デフォルトを少し増やす
    p.add_argument("--max_steps", type=int, default=1000, help="実行ステップ数")
    return p.parse_args()


def main():
    """メイン関数: 環境を起動し、ゼロアクションで実行する"""
    args = _parse_args()

    # Kit/PhysX の起動
    app = AppLauncher(headless=args.headless).app
    try:
        # Cfg の生成 (num_envs を引数で上書き)
        cfg = PorcaroRLEnvCfg() # クラス名を変更
        cfg.scene.num_envs = args.num_envs

        # 環境の生成
        env = PorcaroRLEnv(cfg) # クラス名を変更

        # リセット
        print("[INFO] 環境をリセットします...")
        _ = env.reset()
        print("[INFO] リセット完了。")

        # ゼロアクションで回す（スポーン確認＆動作確認）
        print(f"[INFO] {args.max_steps} ステップのシミュレーションを開始します (ゼロアクション)。")
        for i in range(args.max_steps):
            # アクション空間の次元 (cfg.action_space) に基づいてゼロテンソルを作成
            actions = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
            
            # ステップ実行
            obs, rew, terminated, truncated, info = env.step(actions)
            
            # もし終了/タイムアウトした環境があればリセット
            if (terminated | truncated).any():
                env.reset_done(terminated | truncated)
                
            if i % 100 == 0:
                print(f"  ステップ {i}/{args.max_steps} 実行中...")


    except Exception as e:
        print(f"[ERROR] シミュレーション中にエラーが発生しました: {e}")
    finally:
        
        # --- 変更: ログマネージャ経由で保存 ---
        # 'env' がローカル変数に存在し、logging_manager を持っているか確認
        if 'env' in locals() and hasattr(env, 'logging_manager') and env.logging_manager is not None:
            print("[INFO] シミュレーション終了。ログをファイルに保存します...")
            # env.logger.save() # <-- 古いコード (コメントアウトまたは削除)
            env.logging_manager.save_on_exit() # <-- こちらを使用
        
        print("[INFO] シミュレーション終了。アプリケーションを閉じます。")
        app.close()


if __name__ == "__main__":
    main()