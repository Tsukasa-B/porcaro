"""
Porcaro Robot: Multi-Model Manual Verification Script (Sim-to-Real Ready)
Modified for Replay Verification with Auto-Path Resolution & No-Drum Option
(Fixed: Disabling BOTH Terminations and Timeouts, and Syncing Replay Time)
"""
import argparse
import sys
import os

# -----------------------------------------------------------------------------
# 1. 引数定義 (ライブラリインポート前に解析が必要)
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Multi-Model Verification Script")

# --- 実験モード設定 ---
parser.add_argument("--model_type", type=str, default="A", choices=["A", "B", "C"], 
                    help="Model Type: A (Ideal/Delay), B (Hysteresis), C (ActuatorNet)")
parser.add_argument("--dr", action="store_true", help="Enable Domain Randomization (DR)")
parser.add_argument("--no_drum", action="store_true", help="Disable drum physics (Air Drumming)")

# --- リプレイ設定 ---
parser.add_argument("--replay_csv", type=str, default=None, 
                    help="Filename of real data CSV (searches in external_data/jetson_project)")

# --- 入力信号設定 ---
parser.add_argument("--mode", type=str, default="double", choices=["sine", "step", "const", "double"], 
                    help="Input signal pattern (ignored if --replay_csv is set)")
parser.add_argument("--bpm", type=float, default=120.0, help="Beats Per Minute for 'double' mode")
parser.add_argument("--pattern", type=str, default="1,1,0,0", help="Rhythm pattern (1=Hit, 0=Rest) for 'double' mode")

# --- 圧力/アクション設定 ---
parser.add_argument("--pressure_high", type=float, default=0.55, help="Active pressure [MPa]")
parser.add_argument("--pressure_low", type=float, default=0.05, help="Inactive pressure [MPa]")
parser.add_argument("--pressure_grip", type=float, default=0.3, help="Grip pressure [MPa]")
parser.add_argument("--duty_cycle", type=float, default=0.5, help="Hit duration ratio")

# --- Sine/Step 詳細設定 ---
parser.add_argument("--freq", type=float, default=1.0, help="Frequency for Sine [Hz]")
parser.add_argument("--amp", type=float, default=0.25, help="Amplitude for Sine")
parser.add_argument("--offset", type=float, default=0.3, help="Offset for Sine")
parser.add_argument("--step_interval", type=float, default=1.0, help="Interval for Step [s]")

# --- 環境設定 ---
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--headless", action="store_true", help="Run without GUI")

args, unknown = parser.parse_known_args()

# -----------------------------------------------------------------------------
# 2. App起動 (ここが最重要！他のライブラリより先に実行)
# -----------------------------------------------------------------------------
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# 3. その他ライブラリのインポート (App起動後に移動)
# -----------------------------------------------------------------------------
import traceback
import torch
import numpy as np
import math
import pandas as pd
from scipy.interpolate import interp1d

try:
    from porcaro_rl.tasks.direct.porcaro_rlv1.porcaro_rl_env import PorcaroRLEnv
    from porcaro_rl.tasks.direct.porcaro_rlv1.porcaro_rl_env_cfg import (
        PorcaroRLEnvCfg_ModelA, PorcaroRLEnvCfg_ModelA_DR,
        PorcaroRLEnvCfg_ModelB, PorcaroRLEnvCfg_ModelB_DR,
        PorcaroRLEnvCfg_ModelC, PorcaroRLEnvCfg_ModelC_DR
    )
except Exception as e:
    print(f"[CRITICAL ERROR] Import failed: {e}")
    simulation_app.close()
    sys.exit(1)


# -----------------------------------------------------------------------------
# 4. エージェントクラス (時系列補間機能付き・修正版)
# -----------------------------------------------------------------------------
class SignalGeneratorAgent:
    def __init__(self, num_envs, dt, device, args):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.args = args
        self.t = 0.0
        self.P_MAX = 0.6 
        
        self.is_replay = args.replay_csv is not None
        self.replay_finished = False

        if self.is_replay:
            self._init_replay(args.replay_csv)
        else:
            self._init_pattern_mode(args)

    def _init_replay(self, csv_filename):
        # ファイル探索
        candidates = [
            csv_filename,
            os.path.join("external_data", "jetson_project", csv_filename),
            os.path.join("porcaro_rl", "external_data", "jetson_project", csv_filename),
            os.path.join("source", "porcaro_rl", "external_data", "jetson_project", csv_filename),
            os.path.join("..", "external_data", "jetson_project", csv_filename)
        ]
        final_path = None
        for path in candidates:
            if os.path.exists(path):
                final_path = path
                break
        
        if final_path is None:
            print(f"\n[ERROR] CSV File not found: {csv_filename}")
            sys.exit(1)

        print(f"\n[Agent] REPLAY MODE: Loading {final_path} ...")
        try:
            df = pd.read_csv(final_path)
            
            # --- 【修正】データのサニタイズ（時系列異常の防止） ---
            df = df.sort_values(by='timestamp')
            df = df.drop_duplicates(subset='timestamp', keep='first')
            
            # タイムスタンプを0始まりに正規化
            t_real = df['timestamp'].values
            t_real = t_real - t_real[0]
            
            # 指令値データの抽出
            cmd_data = df[['cmd_DF', 'cmd_F', 'cmd_G']].values

            # --- 【修正】 線形補間関数の作成 ---
            self.replay_interpolator = interp1d(
                t_real, cmd_data, axis=0, kind='linear', 
                bounds_error=False, fill_value=(cmd_data[0], cmd_data[-1])
            )
            self.max_replay_time = t_real[-1]
            print(f" -> Loaded {len(df)} rows. Duration: {self.max_replay_time:.2f}s")
            
        except Exception as e:
            print(f"[Agent] Error loading CSV: {e}")
            sys.exit(1)

    def _init_pattern_mode(self, args):
        self.pattern_seq = [int(x) for x in args.pattern.split(",")]
        self.beat_sec = 60.0 / args.bpm
        self.note_sec = self.beat_sec / 4.0 
        self.total_cycle_sec = self.note_sec * len(self.pattern_seq)
        print(f"\n[Agent] Mode: {args.mode}")

    def _pressure_to_action(self, p_mpa):
        # 圧力(0~0.6) を Action(-1~1) に変換
        p = np.clip(p_mpa, 0.0, self.P_MAX)
        return 2.0 * (p / self.P_MAX) - 1.0

    def get_action(self, obs=None):
        p_df, p_f, p_g = 0.0, 0.0, 0.0
        
        if self.is_replay:
            if self.t > self.max_replay_time + 0.5:
                self.replay_finished = True
                return torch.zeros((self.num_envs, 3), device=self.device)
            
            # 補間の実行
            cmds = self.replay_interpolator(self.t)
            p_df, p_f, p_g = cmds[0], cmds[1], cmds[2]

        elif self.args.mode == "double":
            t_in_cycle = self.t % self.total_cycle_sec
            note_idx = int(t_in_cycle / self.note_sec)
            t_in_note = t_in_cycle % self.note_sec
            if note_idx >= len(self.pattern_seq): note_idx = 0
            is_hit = self.pattern_seq[note_idx] == 1
            target_f = self.args.pressure_low
            target_df = self.args.pressure_high
            if is_hit:
                if t_in_note < (self.note_sec * self.args.duty_cycle):
                    target_f = self.args.pressure_high
                    target_df = self.args.pressure_low
            p_f = target_f; p_df = target_df; p_g = self.args.pressure_grip

        elif self.args.mode == "sine":
            sine_val = math.sin(2 * math.pi * self.args.freq * self.t)
            base = self.args.offset + self.args.amp * sine_val
            p_df = base
            p_f = self.args.offset - (base - self.args.offset)
            p_g = self.args.pressure_grip
            
        elif self.args.mode == "step":
            is_high = int(self.t / self.args.step_interval) % 2 == 0
            val = self.args.pressure_high if is_high else self.args.pressure_low
            p_df = self.args.pressure_low; p_f = val; p_g = self.args.pressure_grip
            
        elif self.args.mode == "const":
            p_df = self.args.offset; p_f = self.args.offset; p_g = self.args.pressure_grip

        actions = torch.zeros((self.num_envs, 3), device=self.device)
        actions[:, 0] = self._pressure_to_action(p_df)
        actions[:, 1] = self._pressure_to_action(p_f)
        actions[:, 2] = self._pressure_to_action(p_g)
        
        self.t += self.dt
        return actions

# -----------------------------------------------------------------------------
# 5. メイン処理
# -----------------------------------------------------------------------------
def main():
    env = None
    try:
        cfg_map = {
            ("A", False): PorcaroRLEnvCfg_ModelA, ("A", True): PorcaroRLEnvCfg_ModelA_DR,
            ("B", False): PorcaroRLEnvCfg_ModelB, ("B", True): PorcaroRLEnvCfg_ModelB_DR,
            ("C", False): PorcaroRLEnvCfg_ModelC, ("C", True): PorcaroRLEnvCfg_ModelC_DR,
        }
        target_cfg_cls = cfg_map.get((args.model_type, args.dr))
        if target_cfg_cls is None: raise ValueError("Invalid Model/DR combination.")
            
        print(f"\n[System] Selecting Config: {target_cfg_cls.__name__}")
        
        env_cfg = target_cfg_cls()
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.controller.control_mode = "pressure" 
        
        # -------------------------------------------------------------
        # [タイムアウト無効化] 
        # -------------------------------------------------------------
        print("[System] Extending episode length to avoid timeout reset.")
        env_cfg.episode_length_s = 1000.0
        
        if hasattr(env_cfg, "terminations"):
            env_cfg.terminations = None
            
        # --- ドラム退避 (No Drum) ---
        if args.no_drum:
            if hasattr(env_cfg, "drum_cfg"):
                env_cfg.drum_cfg.init_state.pos = (0.0, 0.0, -10.0)
            elif hasattr(env_cfg.scene, "drum"):
                env_cfg.scene.drum.init_state.pos = (0.0, 0.0, -10.0)

        # --- ログ設定 ---
        dr_str = "DR" if args.dr else "Ideal"
        if args.replay_csv:
            base_csv = os.path.splitext(os.path.basename(args.replay_csv))[0]
            log_name = f"verify_Model{args.model_type}_{dr_str}_REPLAY_{base_csv}.csv"
        else:
            log_name = f"verify_Model{args.model_type}_{dr_str}_{args.mode}_{int(args.bpm)}bpm.csv"
        
        if hasattr(env_cfg, "logging"):
            env_cfg.logging.enabled = True
            env_cfg.logging.filepath = log_name

        print(f"[System] Log file: {log_name}")

        env = PorcaroRLEnv(cfg=env_cfg)
        
        # ---------------------------------------------------------------------
        # 【修正】 DT計算ロジック： デシメーションズレを防止
        # ---------------------------------------------------------------------
        # 1. DirectRLEnvの step_dt プロパティがあればそれを信じる
        if hasattr(env, "step_dt"):
             dt_step = env.step_dt
             print(f"[System] Using env.step_dt: {dt_step:.5f}s")
        else:
             # 2. なければ計算。ただし decimation が 1 になっている可能性を警戒
             sim_dt = env.cfg.sim.dt
             decimation = getattr(env.cfg, "decimation", 4) # Default to 4 if missing
             dt_step = sim_dt * decimation
             print(f"[System] Calculated dt: {dt_step:.5f}s (sim_dt={sim_dt} * dec={decimation})")

        # 【安全策】 dt_step が極端に小さい(0.01未満)場合は、制御周波数50Hz(0.02s)に強制補正
        # 「波形が4倍伸びる」現象はここが 0.005s になっているのが原因
        if dt_step < 0.01:
            print(f"\n[WARNING] dt_step ({dt_step:.5f}s) is too small! Likely decimation mismatch.")
            print(f"[WARNING] Forcing dt_step = 0.02s (50Hz) to sync with real robot data.")
            dt_step = 0.02

        agent = SignalGeneratorAgent(env.num_envs, dt_step, env.device, args)
        
        obs, _ = env.reset()
        print("[System] Simulation Loop Started...")

        while simulation_app.is_running():
            actions = agent.get_action(obs)
            
            if agent.is_replay and agent.replay_finished:
                print("[System] Replay finished.")
                break

            obs, rew, terminated, truncated, info = env.step(actions)
            
    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        traceback.print_exc()
    finally:
        if env is not None:
            print("\n[System] Closing environment and saving logs...")
            env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()