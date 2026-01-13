# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/logging/logging_manager.py
from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Any, List, Sequence
import traceback

if TYPE_CHECKING:
    from ..porcaro_rl_env import PorcaroRLEnv
    from .datalogger import DataLogger

class LoggingManager:
    def __init__(self, 
                 env: PorcaroRLEnv, 
                 dt: float, 
                 log_filepath: str | None = None,
                 enable_logging: bool = False):
        
        self.env = env
        self.device = env.device
        self.dt = dt
        self.num_envs = env.num_envs
        self.enable_logging = enable_logging
        self.current_time_s = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        self.logger: DataLogger | None = None
        if enable_logging and log_filepath is not None:
            # ヘッダー定義
            headers = [
                "time_s",
                "action_0", "action_1", "action_2",
                "q_wrist_deg", "q_grip_deg",
                "qd_wrist_deg", "qd_grip_deg",
                "force_z", "f1_score",
                "target_force", # ターゲットの波形 (0~20N)
                "target_bpm",   # 現在のBPM
                "P_cmd_DF", "P_cmd_F", "P_cmd_G",
                "P_out_DF", "P_out_F", "P_out_G",
                "tau_wrist", "tau_grip"
            ]
            from .datalogger import DataLogger
            self.logger = DataLogger(log_filepath, headers=headers)
            print(f"[LoggingManager] Logging enabled -> {log_filepath}")

        self.step_data_buffer: List[List[float]] = []
        # デバッグ用: 最初の1回だけデータ受信確認ログを出すフラグ
        self._debug_print_done = False 

    def reset_idx(self, env_ids: torch.Tensor | Sequence[int] | None = None):
        self.step_data_buffer = []
        if env_ids is not None:
            self.current_time_s[env_ids] = 0.0
        else:
            self.current_time_s.fill_(0.0)

    def reset(self):
        self.reset_idx(None)

    def save_on_exit(self):
        """終了時の処理"""
        print("[LoggingManager] Saving logs on exit...")
        if self.logger:
            self.logger.save()
        self.step_data_buffer = []

    def update_time(self, dt: float):
        self.current_time_s += dt

    def buffer_step_data(self, 
                         q_full: torch.Tensor, 
                         qd_full: torch.Tensor, 
                         telemetry: dict | None, 
                         actions: torch.Tensor,
                         current_sim_time: float,
                         target_force: float = 0.0,
                         target_bpm: float = 0.0,
                         ):
        """物理ステップごとに呼ばれ、データをリストに「追記」する"""
        if self.logger is None:
            return

        # デバッグ: データが来ているか確認 (最初の1回のみ表示)
        if not self._debug_print_done:
            print(f"[LoggingManager] Received data at t={current_sim_time:.3f}. Buffering...")
            self._debug_print_done = True

        env_idx = 0 
        
        try:
            t = current_sim_time
            act = actions[env_idx].detach().cpu().tolist()
            
            dof_ids = self.env.dof_idx
            q_deg = torch.rad2deg(q_full[env_idx, dof_ids]).detach().cpu().tolist()
            qd_deg = torch.rad2deg(qd_full[env_idx, dof_ids]).detach().cpu().tolist()
            
            # 変更箇所: テレメトリ取得のロバスト化
            # 値がない場合やActuatorNet使用時は 0.0 (またはNaN) で埋める
            p_cmd = [0.0] * 3
            p_out = [0.0] * 3
            tau_w = 0.0
            tau_g = 0.0

            if telemetry:
                if "P_cmd" in telemetry:
                    p_cmd = telemetry["P_cmd"][env_idx].detach().cpu().tolist()
                if "P_out" in telemetry:
                    p_out = telemetry["P_out"][env_idx].detach().cpu().tolist()
                if "tau_w" in telemetry:
                    tau_w = telemetry["tau_w"][env_idx].item()
                if "tau_g" in telemetry:
                    tau_g = telemetry["tau_g"][env_idx].item()

            # 構成: Time(1) + Act(3) + Q(2) + Qd(2) + ForcePlaceholder(2) + Target(2) + P_cmd(3) + P_out(3) + Tau(2)
            row = [t] + act + q_deg + qd_deg + [0.0, 0.0] + [target_force, target_bpm] + p_cmd + p_out + [tau_w, tau_g]
            self.step_data_buffer.append(row)
            
        except Exception as e:
            print(f"[LoggingManager] Buffer Error: {e}")
            traceback.print_exc()

    def finalize_log_step(self, peak_force: torch.Tensor, f1_force: torch.Tensor, step_reward: torch.Tensor):
        """エージェントステップの終わりに呼ばれる"""
        if self.logger is None:
            self.step_data_buffer = []
            return

        if not self.step_data_buffer:
            return

        env_idx = 0
        try:
            rows_to_write = []
            num_history = len(self.step_data_buffer)
            
            force_history = peak_force[env_idx].detach().cpu()
            if force_history.numel() > 0:
                if force_history.dim() > 0:
                    force_history = torch.flip(force_history, dims=[0])
            
            f1_val = f1_force[env_idx].item()

            for i in range(num_history):
                # buffer_step_dataで作った行をコピー
                final_row = list(self.step_data_buffer[i])
                
                # --- Forceの取得 ---
                f_val = 0.0
                if force_history.numel() > 0 and i < force_history.shape[0]:
                    current_force = force_history[i]
                    flat_force = current_force.flatten()
                    if flat_force.numel() >= 3:
                        f_val = flat_force[2].item() # Z軸
                    elif flat_force.numel() > 0:
                        f_val = flat_force[0].item()

                # ★修正: プレースホルダ (インデックス8, 9) を上書きするだけにする
                # [0]:Time, [1-3]:Act, [4-5]:Q, [6-7]:Qd, [8]:Force, [9]:F1, [10]:TgtF, [11]:TgtBPM...
                final_row[8] = f_val
                final_row[9] = f1_val
                
                rows_to_write.append(final_row)

            if rows_to_write:
                self.logger.add_data_batch(rows_to_write)
                self.logger.save()

        except Exception as e:
            print(f"[LoggingManager] Finalize Error: {e}")
            traceback.print_exc()
        
        finally:
            self.step_data_buffer = []