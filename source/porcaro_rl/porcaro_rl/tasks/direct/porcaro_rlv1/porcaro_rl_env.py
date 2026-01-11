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
from isaaclab.managers import EventManager # <--- 追加

# porcaro_rl_env.py のインポートセクション

# パッケージ実行時
from .porcaro_rl_env_cfg import PorcaroRLEnvCfg
from .actions.base import ActionController # IF (型ヒント用)
from .actions.torque import TorqueActionController
from .logging.logging_manager import LoggingManager # <-- 新規追加
from .rewards.reward import RewardManager          # <-- 新規追加
from .cfg.rewards_cfg import RewardsCfg # 明示的にインポート
from .rhythm_generator import RhythmGenerator # <--- 新規追加



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
        # dt_ctrl = sim.dt
        dt_ctrl = self.cfg.sim.dt # 物理エンジンのステップ数cfgで設定したものと同じ
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
        
        # --- 追加: コントローラの状態をリセット ---
        # (reset() は device と n_envs を要求するため __init__ の最後で呼ぶ)
        self.action_controller.reset(self.num_envs, self.device)

        # 1. ロギングマネージャ
        self.logging_manager = LoggingManager(
            env=self,                            # ★これが最重要！環境自体を渡す
            dt=self.cfg.sim.dt,                  # 物理ステップ時間 (0.005s)
            log_filepath=self.cfg.logging.filepath, 
            enable_logging=self.cfg.logging.enabled
        )

        # print(f"[DEBUG] ロギングを強制的に有効化します (Config上: {self.cfg.logging.enabled})")
        # self.logging_manager.force_enable()
        # # (追加) 報酬ログも強制有効化
        # print(f"[DEBUG] 報酬ログを強制的に有効化します (Config上: {self.cfg.reward_logging.enabled})")
        # self.logging_manager.force_enable_reward_log()
        
        # 2. 報酬マネージャ
        self.reward_manager = RewardManager(
            cfg=self.cfg.rewards,
            num_envs=self.num_envs,
            device=self.device,
        )

        # target_bpm が Config にない場合は、bpm_max またはデフォルト値(120)を使う
        self.bpm = getattr(self.cfg.rewards, "target_bpm", getattr(self.cfg.rewards, "bpm_max", 120.0))
        self.beat_interval = 60.0 / self.bpm if self.bpm > 0 else 0.5

        # --- ▼▼▼ 追加項目: リズム生成器の初期化 ▼▼▼ ---
        # エラー防止: cfg内に必要なパラメータがあるか確認し、無ければデフォルト値を使う安全策を入れる
        bpm_range = (
            getattr(self.cfg.rewards, "bpm_min", 60.0),
            getattr(self.cfg.rewards, "bpm_max", 160.0)
        )
        prob_rest = getattr(self.cfg.rewards, "prob_rest", 0.3)
        prob_double = getattr(self.cfg.rewards, "prob_double", 0.2)

        # 制御周期の計算 (Sim dt * decimation)
        self.dt_ctrl = self.cfg.sim.dt * self.cfg.decimation

        self.rhythm_generator = RhythmGenerator(
            num_envs=self.num_envs,
            device=self.device,
            dt=self.dt_ctrl,
            max_episode_length=self.max_episode_length,
            bpm_range=bpm_range,
            prob_rest=prob_rest,
            prob_double=prob_double
        )
        
        # 先読みステップ数の計算
        lookahead_horizon = getattr(self.cfg, "lookahead_horizon", 0.5) # cfgに無い場合は0.5秒
        self.lookahead_steps = int(lookahead_horizon / self.dt_ctrl)
        
        # [安全確認] 観測空間の次元チェック
        # Joint Pos(2) + Vel(2) + BPM(1) + Buffer(lookahead_steps)
        expected_dim = 2 + 2 + 1 + self.lookahead_steps
        if self.cfg.observation_space != expected_dim:
            print(f"[WARNING] 観測空間の次元不一致の可能性があります。")
            print(f"  Config設定値: {self.cfg.observation_space}")
            print(f"  計算上の必要数: {expected_dim}")
            print("  -> porcaro_rl_env_cfg.py の observation_space を修正してください。")
        # -----------------------------------------------

        # ▼▼▼ 追加: 診断用プリント（ここを入れてください！） ▼▼▼
        print("=" * 80)
        print(f"[DEBUG DIAGNOSTIC] 環境設定値の確認")
        print(f"  - sim.dt (物理ステップ): {self.cfg.sim.dt}")
        print(f"  - decimation (間引き数): {self.cfg.decimation}")
        
        actual_dt_ctrl = self.cfg.sim.dt * self.cfg.decimation
        print(f"  - dt_ctrl (制御周期)   : {actual_dt_ctrl:.5f} 秒")
        print("=" * 80)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        if self.cfg.events is not None:
            self.event_manager = EventManager(self.cfg.events, self)
            print("[INFO] Event Manager initialized. Applying startup randomizations...")
            self.event_manager.apply(mode="startup") # 質量などのランダム化を実行
        else:
            self.event_manager = None

        print("[INFO] Initializing rhythm patterns for all environments...")
        all_ids = torch.arange(self.num_envs, device=self.device)
        self.rhythm_generator.reset(all_ids)

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

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """親クラスのstepを呼び出した後に、難易度調整の処理を挟む"""
        
        # 1. まず親クラス(DirectRLEnv)の本来の処理を行わせる
        obs, rew, terminated, truncated, extras = super().step(actions)

        # 2. ここに「割り込み」処理を書く (カリキュラム更新)
        if hasattr(self, "common_step_counter"):
            # 学習が進むにつれ difficulty を 0.0 -> 1.0 に上げる
            # 例: 1億ステップ(全環境合計ではなくシム内時間)でMAXにする
            # 50Hz * 2000秒 = 100,000 ステップくらいを目安にする
            
            # 簡易的に common_step_counter (物理ステップ数) を使う
            # 1回のstep呼び出しで common_step_counter は decimation分進む
            
            # 難易度が1.0になるまでのステップ数 (調整してください)
            # ここでは「学習完了の8割くらいの時間」を設定するのが一般的
            TOTAL_TRAINING_STEPS = 100_000 # ※後述の計算に基づく目安
            
            difficulty = min(self.common_step_counter / TOTAL_TRAINING_STEPS, 1.0)
            
            # リズム生成器に反映
            if hasattr(self, "rhythm_generator"):
                self.rhythm_generator.set_difficulty(difficulty)
                
            # ログに残す
            extras["Episode/difficulty"] = torch.tensor(difficulty)

        return obs, rew, terminated, truncated, extras

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
        current_steps = self.episode_length_buf
        # ターゲット力 (Tensorなので .item() でfloatにする)
        tgt_val = self.rhythm_generator.get_current_target(current_steps)[0].item()
        # 現在のBPM
        tgt_bpm = self.rhythm_generator.current_bpms[0].item()

        # --- 変更: データロギング (マネージャに委譲) ---
        
        # 内部時刻を更新
        self.logging_manager.update_time(self.cfg.sim.dt)
        
        # (変更) f1 を渡さず、buffer_step_data を呼ぶ
        self.logging_manager.buffer_step_data(
            q_full=self.robot.data.joint_pos,
            qd_full=self.robot.data.joint_vel,
            telemetry=self.action_controller.get_last_telemetry(),
            actions=self.actions,
            current_sim_time=self.sim.current_time, # ★ここを追加！(物理時間を渡す)
            target_force=tgt_val, # ★ここ！
            target_bpm=tgt_bpm
        )

    def _get_observations(self) -> dict:
        """環境の観測を取得する"""
        
        # 1. 物理状態 (既存)
        q = self.robot.data.joint_pos[:, self.dof_idx]   # [num_envs, 2]
        qd = self.robot.data.joint_vel[:, self.dof_idx]  # [num_envs, 2]
        
        # --- ▼▼▼ 変更項目: リズム情報の取得 ▼▼▼ ---
        
        # (削除) 以前のサイン波やターゲット時刻の計算ロジックは削除します。
        
        # (追加) 現在のステップ数
        current_steps = self.episode_length_buf
        
        # (A) 現在のBPM情報
        # squeeze/unsqueeze周りで事故らないよう、形状を明示
        bpm_obs = self.rhythm_generator.current_bpms.view(self.num_envs, 1) / 180.0
        
        # (B) 先読みバッファ [num_envs, lookahead_steps]
        rhythm_buf = self.rhythm_generator.get_lookahead(current_steps, self.lookahead_steps)
        
        # 値の正規化: ターゲット力(例: 20N)で割って 0~1 にする
        # ※ cfgから取るか、マジックナンバー回避で定数を使う
        target_force = getattr(self.cfg.rewards, "target_force_fd", 20.0)
        rhythm_buf = rhythm_buf / target_force

        # 結合: [Pos(2), Vel(2), BPM(1), Buffer(N)]
        obs = torch.cat((q, qd, bpm_obs, rhythm_buf), dim=-1)
        # -----------------------------------------
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """報酬を計算する"""

        # (A) センサーデータ取得 (既存)
        force_history_tensor = self.stick_sensor.data.net_forces_w_history
        current_rl_step_force_history = force_history_tensor[:, 0:self.cfg.decimation, :]
        force_z_raw = current_rl_step_force_history[..., 2]
        force_z_history_in_step = force_z_raw.reshape(self.num_envs, self.cfg.decimation)

        # 時刻情報
        current_time = self.logging_manager.current_time_s.reshape(-1)
        dt_sim = self.cfg.sim.dt

        # --- 既存の RewardManager の呼び出し (ログ/Done判定用) ---
        # 報酬(old_rewards)は使わないが、内部状態更新のために呼んでおく
        _ = self.reward_manager.compute_reward_and_dones(
            force_z_history=force_z_history_in_step,
            current_time_s=current_time,
            dt_step=dt_sim,
            episode_length_buf=self.episode_length_buf,
            max_episode_length=(self.max_episode_length - 1) 
        )
        
        # --- ▼▼▼ 追加項目: 新しい報酬計算ロジック ▼▼▼ ---
        
        # 1. 現在のターゲット値を取得
        current_steps = self.episode_length_buf
        target_val = self.rhythm_generator.get_current_target(current_steps) # (num_envs,)
        
        # 2. 実測値 (このステップ内の最大打撃力)
        # 負の値(ノイズ)を除外するため clamp(min=0)
        current_force_max = torch.max(force_z_history_in_step, dim=1).values.clamp(min=0.0)
        
        # 3. マッチング報酬 (Timing & Force)
        # ターゲットが 1.0N 以上のときだけ評価
        is_hit_target = target_val > 1.0
        
        force_error = current_force_max - target_val
        sigma_force = getattr(self.cfg.rewards, "sigma_force", 10.0)
        
        # ガウスカーネルで類似度計算 (差が0なら1.0, 離れると減る)
        rew_match = torch.exp(- (force_error**2) / (sigma_force**2))
        
        # ターゲットが無い区間はマッチング報酬をゼロにする
        rew_match = torch.where(is_hit_target, rew_match, torch.zeros_like(rew_match))
        
        # 4. 休符ペナルティ (Rest Violation)
        # ターゲットが小さい(休符)のに、力が閾値を超えている場合
        force_threshold_rest = getattr(self.cfg.rewards, "force_threshold_rest", 1.0)
        is_rest_target = target_val <= 1.0
        violation = (current_force_max > force_threshold_rest)
        
        rew_rest_penalty = torch.zeros_like(rew_match)
        # 違反したら力の大きさに比例して減点
        rew_rest_penalty = torch.where(
            is_rest_target & violation,
            -1.0 * current_force_max, 
            torch.zeros_like(rew_match)
        )

        # 5. 合計報酬
        weight_match = getattr(self.cfg.rewards, "weight_match", 20.0)
        weight_rest = getattr(self.cfg.rewards, "weight_rest_penalty", 10.0)
        
        total_reward = (weight_match * rew_match) + (weight_rest * rew_rest_penalty)
        
        # ---------------------------------------------------
        
        # ログ書き込み (LoggingManager)
        f1_force_tensor = self.reward_manager.get_first_hit_force() # 既存計算を利用
        self.logging_manager.finalize_log_step(
            peak_force=force_history_tensor,
            f1_force=f1_force_tensor,
            step_reward=total_reward # 新しい報酬をログに残す
        )
        
        # Done判定は RewardManager (または親クラス) のものを利用
        self.reset_terminated, self.reset_time_outs = self.reward_manager.get_dones()
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """報酬計算ステップで計算済みの終了判定を返す"""
        
        # --- 変更: RewardManager が計算した値を返す ---
        return self.reset_terminated, self.reset_time_outs

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """指定された環境IDの内部状態をリセットします。"""
        
        # 1. 親クラスのリセット (ロボット姿勢など)
        super()._reset_idx(env_ids)
        
        # 2. 既存マネージャのリセット
        if hasattr(self, "event_manager") and self.event_manager is not None:
            self.event_manager.reset(env_ids)
        if hasattr(self, "reward_manager"):
            self.reward_manager.reset_idx(env_ids)
        if hasattr(self, "logging_manager"):
            self.logging_manager.reset_idx(env_ids)
            
        # --- ▼▼▼ 追加項目: リズムパターンの再生成 ▼▼▼ ---
        if hasattr(self, "rhythm_generator"):
            self.rhythm_generator.reset(env_ids)
        # -----------------------------------------------

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