# logging/logging_manager.py
from __future__ import annotations
import torch
from typing import Sequence, List, Any

# --- 必要なモジュールをインポート ---

from ..cfg.logging_cfg import LoggingCfg, RewardLoggingCfg
from .datalogger import DataLogger

class LoggingManager:
    """
    シミュレーションのデータロギングを管理するクラス。
    「何を」「どの順序で」ログに記録するかを責務として持つ。
    ファイルI/O自体は DataLogger に委譲する。
    """
    def __init__(self, 
                 cfg: LoggingCfg, 
                 reward_cfg: RewardLoggingCfg,
                 dof_idx: Sequence[int], 
                 num_envs: int, 
                 device: str | torch.device,
                 dt: float,                  # <-- (新規追加)
                 decimation: int,
                 ):
        
        # --- ここから追加 (引数を self に保存) ---
        self.cfg = cfg
        self.reward_cfg = reward_cfg
        self.dof_idx = dof_idx
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dt = dt                          # <-- (新規追加) 
        self.decimation = decimation
        # --- ここまで追加 ---
        
        self.logger: DataLogger | None = None
        self.reward_logger: DataLogger | None = None
        
        # Cfg に基づいてロガーを初期化 (ただし force_enable() で上書き可能)
        if cfg.enabled:
            print(f"[LoggingManager] ロギングを有効化します (Config: {cfg.enabled})")
            self.logger = DataLogger(
                filepath=cfg.filepath,
                headers=cfg.headers
            )
        else:
            print(f"[LoggingManager] Cfg上はロギングが無効です (Config: {cfg.enabled})")

        if reward_cfg.enabled:
            print(f"[LoggingManager] 報酬ログを有効化します (Config: {reward_cfg.enabled})")
            self.reward_logger = DataLogger(
                filepath=reward_cfg.filepath,
                headers=reward_cfg.headers
            )
        else:
            print(f"[LoggingManager] Cfg上は報酬ログが無効です (Config: {reward_cfg.enabled})")

        # env.py から移動した変数
        self.current_time_s = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.log_step_counter: int = 0
        # (この間隔は将来的に LoggingCfg に移しても良い)
        self.log_save_interval: int = 2000 

        self.step_data_buffer: List[Any] | None = None

    def force_enable(self):
        """
        Cfg の設定 (self.cfg.enabled) に関わらず、
        DataLogger を強制的に初期化する。
        """
        if self.logger is None:
            print(f"[LoggingManager] ロギングを強制的に有効化します。")
            self.logger = DataLogger(
                filepath=self.cfg.filepath,
                headers=self.cfg.headers
            )

    def force_enable_reward_log(self):
        """報酬ロガーを強制的に有効化"""
        if self.reward_logger is None:
            print(f"[LoggingManager] 報酬ログを強制的に有効化します。")
            self.reward_logger = DataLogger(
                filepath=self.reward_cfg.filepath,
                headers=self.reward_cfg.headers
            )

    def update_time(self, dt_ctrl: float):
        """
        内部のシミュレーション時刻を更新する。
        (env._apply_action のタイミングで呼ばれることを想定)
        """
        self.current_time_s.add_(dt_ctrl)

    def reset_idx(self, env_ids: Sequence[int]):
        """
        指定された環境の内部状態（時刻）をリセットする。
        (env._reset_idx のタイミングで呼ばれることを想定)
        """
        self.current_time_s[env_ids] = 0.0
    
    def buffer_step_data(self, 
                         q_full: torch.Tensor,       # robot.data.joint_pos
                         qd_full: torch.Tensor,      # robot.data.joint_vel
                         telemetry: dict | None,      # action_controller.get_last_telemetry()
                         actions: torch.Tensor
                         ):
        """
        (変更) ログデータをCSVに書き込まず、一時バッファに保存する。
        (env._apply_action から呼ばれる)
        """
        
        if self.logger is None:
            self.step_data_buffer = None # ロガーが無効ならバッファもクリア
            return

        env_idx = 0 # 0番目の環境のみログ取得
        
        try:
            # 1. 時間
            time_s_tensor = self.current_time_s[env_idx].reshape(-1) # <-- 修正

            # actions[0] を取得し、(3,) の形にする
            act = actions[env_idx].reshape(-1)
            
            # 2. 関節角度・速度
            q = q_full[env_idx, self.dof_idx].reshape(-1)      # (2,) <-- 修正
            qd = qd_full[env_idx, self.dof_idx].reshape(-1)    # (2,) <-- 修正
            
            
            # 4. コントローラ内部状態 (テレメトリから取得)
            if telemetry:
                p_cmd = telemetry["P_cmd"][env_idx].reshape(-1) # (3,) <-- 修正
                p_out = telemetry["P_out"][env_idx].reshape(-1) # (3,) <-- 修正
                tau_w = telemetry["tau_w"][env_idx].reshape(-1) # (1,) <-- 修正
                tau_g = telemetry["tau_g"][env_idx].reshape(-1) # (1,) <-- 修正
            else:
                p_cmd = torch.zeros(3, device=self.device)
                p_out = torch.zeros(3, device=self.device)
                tau_w = torch.zeros(1, device=self.device)
                tau_g = torch.zeros(1, device=self.device)

            self.step_data_buffer = [
                time_s_tensor, # time_s
                act,           # act_theta_eq, act_K_wrist, act_K_grip (<-- ここに追加)
                q,             # q_wrist, q_grip
                qd,            # qd_wrist, qd_grip
                p_cmd,         # P_cmd...
                p_out,         # P_out...
                tau_w,         # tau_w
                tau_g,         # tau_g
            ]
            
        except Exception as e:
            print(f"[WARN] ロギングデータのバッファリング中にエラー: {e}")
            self.step_data_buffer = None

    def finalize_log_step(self, 
                          peak_force: torch.Tensor,  # (これは net_forces_w_history 全体)
                          f1_force: torch.Tensor,      
                          step_reward: torch.Tensor,  
                          ):
        """
        (変更) 物理ステップ(dt)ごとに、decimation回ループしてログを書き込む
        (env._get_rewards から呼ばれる)
        """
        env_idx = 0 # 0番目の環境のみログ取得

        # --- 1. シミュレーションログ (f1 を追加して書き込み) ---
        if self.logger is not None and self.step_data_buffer is not None:
            try:
                rows_to_log: list[list[float]] = []
                # バッファリング済みのデータ（q, qd, p_cmd, p_out, tau）を取得
                # これらは物理ステップ「前」の値であり、ループ中使い回す
                base_data_tensors = self.step_data_buffer
                
                # バッファリングされた時間（エージェントステップの終了時間 T_end）
                time_s_end = base_data_tensors[0] # (1,)
                
                # f1_force（エージェントステップで確定した値）も使い回す
                f1 = f1_force[env_idx].reshape(-1) # (1,)
                
                # (変更) decimation 回ループして書き込む
                # 履歴は [0] が最新のため、逆順 (decimation-1 ... 0) でループし、
                # 時刻を dt ずつ計算しながら書き込む
                for i in range(self.decimation):
                    
                    # 履歴インデックス (i=0 -> idx=3, i=1 -> idx=2, ... i=3 -> idx=0) (decimation=4の場合)
                    history_idx = (self.decimation - 1) - i
                    
                    # (A) 時刻の計算
                    # T_end から (decimation - i) * dt を引く
                    # i=0 (idx=3): T_end - 4*dt (このステップの開始時刻)
                    # i=1 (idx=2): T_end - 3*dt
                    # ...
                    # i=3 (idx=0): T_end - 1*dt (このステップの終了直前)
                    time_s_current_step = time_s_end - (self.decimation - i) * self.dt
                    
                    # (B) 物理ステップの力
                    force = peak_force[env_idx, history_idx].reshape(-1) # (3,)
                    
                    # (C) データリストを再構築
                    log_tensors_dt = list(base_data_tensors) # コピー
                    log_tensors_dt[0] = time_s_current_step.reshape(-1) # (A) 時刻を上書き
                    log_tensors_dt.insert(4, force) # (B) 力を挿入
                    log_tensors_dt.append(f1)       # (C) f1 を追加
                
                    data_row_tensor = torch.cat(log_tensors_dt)
                    data_row_list = data_row_tensor.cpu().tolist()
                    
                    rows_to_log.append(data_row_list)

                if rows_to_log:
                    # --- ▼▼▼ 改善点 4: 安全機構(アサーション)の追加 ▼▼▼ ---
                    # 最初の1行の列数とヘッダーの列数が一致しているか確認
                    assert len(rows_to_log[0]) == len(self.logger.headers), \
                        f"[LoggingManager] ERROR: データ列数({len(rows_to_log[0])})が" \
                        f"ヘッダー列数({len(self.logger.headers)})と一致しません！"
                    
                    self.logger.add_data_batch(rows_to_log)

            except Exception as e:
                # (エラーメッセージを具体的に)
                print(f"[WARN] 物理ステップごとのログ書き込み確定中にエラー: {e}")
            
            # バッファをクリア
            self.step_data_buffer = None

        # --- 2. 報酬ログ (log_reward のロジックをここに統合) ---
        if self.reward_logger is not None:
            try:
                # headers: ["time_s", "step_reward"]
                time_s = self.current_time_s[env_idx].cpu().item()
                reward_val = step_reward[env_idx].cpu().item()
                
                self.reward_logger.add_step([time_s, reward_val])

            except Exception as e:
                print(f"[WARN] 報酬データの収集中にエラー: {e}")

        # --- 3. 定期保存 (書き込み確定のタイミングで実行) ---
        self.log_step_counter += 1
        if (self.log_step_counter % self.log_save_interval) == 0:
            print(f"[DataLogger] 定期保存 (ステップ {self.log_step_counter}) を実行します。")
            if self.logger:
                self.logger.save()
            if self.reward_logger:
                self.reward_logger.save()

    def save_on_exit(self):
        """シミュレーション終了時に、バッファに残っているログを保存する"""
        if self.logger is not None:
            print("[INFO] シミュレーションログをファイルに保存します...")
            self.logger.save()
        
        if self.reward_logger is not None:
            print("[INFO] 報酬ログをファイルに保存します...")
            self.reward_logger.save()