"""
Porcaro Robot: IROS 2026 Validation Experiment Runner (Sim)
Target: Isaac Lab (Model A/B)
Author: Robo-Dev Partner

[Directory Structure Assumption]
porcaro/ (Project Root)
  ├── scripts/verification/run_manual_verification_iros.py
  └── external_data/jetson_project/IROS/test_signals/ (CSV files here)

[Usage]
  python scripts/verification/run_manual_verification_iros.py exp1_static_hysteresis
  python scripts/verification/run_manual_verification_iros.py exp1_static_hysteresis --model_type B
  python scripts/verification/run_manual_verification_iros.py exp2_step_response --model_type B
  python scripts/verification/run_manual_verification_iros.py exp3_frequency_sweep --model_type B
  python scripts/verification/run_manual_verification_iros.py data_exp1_static_hysteresis_TIMESTAMP

"""

# =============================================================================
# ★ GUI Crash Fix (Must be first)
# =============================================================================
import matplotlib
matplotlib.use('Agg') 

import argparse
import sys
import os
import glob
import traceback
import math
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. Arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="IROS Validation Runner (Sim)")

# Positional Argument: CSV Name
parser.add_argument("csv_name", nargs="?", type=str, default=None, help="Experiment CSV name (e.g. exp1_static_hysteresis)")

# Options
parser.add_argument("--model_type", type=str, default="B", choices=["A", "B", "C"], help="Model Type (A:Old, B:New, C:Future)")
parser.add_argument("--dr", action="store_true", help="Enable Domain Randomization")
parser.add_argument("--no_drum", action="store_true", default=False, help="Remove drum (Default: True)")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--tau", type=float, default=0.15, help="Pressure Time Constant (tau) for reconstruction [s]")

args, unknown = parser.parse_known_args()

# -----------------------------------------------------------------------------
# 2. Launch Isaac Lab
# -----------------------------------------------------------------------------
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# 3. Imports
# -----------------------------------------------------------------------------
try:
    from porcaro_rl.tasks.direct.porcaro_rlv1.porcaro_rl_env import PorcaroRLEnv
    from porcaro_rl.tasks.direct.porcaro_rlv1.porcaro_rl_env_cfg import (
        PorcaroRLEnvCfg,
        PorcaroRLEnvCfg_ModelA, PorcaroRLEnvCfg_ModelA_DR,
        PorcaroRLEnvCfg_ModelB, PorcaroRLEnvCfg_ModelB_DR,
        PorcaroRLEnvCfg_ModelC, PorcaroRLEnvCfg_ModelC_DR
    )
except Exception as e:
    print(f"[CRITICAL ERROR] Import failed: {e}")
    simulation_app.close()
    sys.exit(1)

# -----------------------------------------------------------------------------
# 4. Agent Class
# -----------------------------------------------------------------------------
class SignalGeneratorAgent:
    def __init__(self, num_envs, dt, device, args):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.args = args
        self.t = 0.0
        self.P_MAX = 0.6 
        
        # Internal Pressure State (for logging)
        # P_new = P_old + (dt/tau)*(P_cmd - P_old)
        self.pres_state = np.zeros(3) # [DF, F, G]
        self.alpha = self.dt / (self.args.tau + self.dt)

        self.is_replay = False
        self.replay_finished = False
        self.csv_path = None
        self.last_cmd = np.zeros(3)

        if args.csv_name:
            self.is_replay = True
            self.csv_path = self._resolve_path(args.csv_name)
            self._init_replay(self.csv_path)
        else:
            print(f"[Agent] No CSV specified. Using Dummy Mode.")

    def _resolve_path(self, name):
        """Find the CSV file recursively"""
        if not name.endswith('.csv'): name += '.csv'
        
        # Priority 1: Direct path
        if os.path.exists(name): return name
        
        # Priority 2: Recursive search in current dir
        files = glob.glob(f"**/{name}", recursive=True)
        if files: return files[0]
        
        # Priority 3: Fuzzy search (keyword match)
        base_name = os.path.splitext(name)[0]
        files = glob.glob(f"**/*{base_name}*.csv", recursive=True)
        if files:
            print(f"[Info] Exact match not found. Using: {files[0]}")
            return files[0]

        print(f"\n[ERROR] CSV File not found: {name}")
        sys.exit(1)

    def _init_replay(self, path):
        print(f"\n[Agent] Loading Replay Sequence: {path}")
        try:
            df = pd.read_csv(path)
            
            # Time column detection
            t_col = next((c for c in ['time', 'timestamp', 'Time'] if c in df.columns), df.columns[0])
            df = df.sort_values(by=t_col)
            t_real = df[t_col].values
            t_real = t_real - t_real[0]

            # Pressure Command Selection
            # 1. Real Measured Pressure (Best for validation)
            if 'meas_pres_DF' in df.columns:
                print(" -> Using REAL MEASURED pressure as Command (Input Realism)")
                cmd_data = df[['meas_pres_DF', 'meas_pres_F', 'meas_pres_G']].values
            # 2. Command Pressure
            elif 'cmd_pressure_DF' in df.columns:
                print(" -> Using COMMAND pressure")
                cmd_data = df[['cmd_pressure_DF', 'cmd_pressure_F', 'cmd_pressure_G']].values
            # 3. Fallback
            elif 'cmd_DF' in df.columns:
                 cmd_data = df[['cmd_DF', 'cmd_F', 'cmd_G']].values
            else:
                raise ValueError("CSV format error: Pressure columns not found.")
            
            # Fill NaN
            cmd_data = np.nan_to_num(cmd_data, nan=0.0)

            self.replay_interpolator = interp1d(
                t_real, cmd_data, axis=0, kind='linear', 
                bounds_error=False, fill_value=(cmd_data[0], cmd_data[-1])
            )
            self.max_replay_time = t_real[-1]
            print(f" -> Duration: {self.max_replay_time:.2f}s")
            
        except Exception as e:
            print(f"[Error] Failed to load CSV: {e}")
            sys.exit(1)

    def get_action(self, obs=None):
        cmd = np.zeros(3) # [DF, F, G]
        
        if self.is_replay:
            if self.t > self.max_replay_time + 1.0:
                self.replay_finished = True
                return torch.zeros((self.num_envs, 3), device=self.device)
            
            cmd = self.replay_interpolator(self.t)
        else:
            cmd = np.array([0.3, 0.3, 0.3])

        self.last_cmd = cmd

        # --- Reconstruct Internal Pressure Dynamics (Sim Logic) ---
        # Update internal pressure state (Low-pass filter)
        self.pres_state = self.pres_state + self.alpha * (cmd - self.pres_state)

        # Normalize Action (0.0~0.6MPa -> -1.0~1.0)
        # Note: We send the *Command* to the Sim. Sim will apply the delay internally.
        # But we also calculate 'pres_state' here just for logging comparison.
        actions = torch.zeros((self.num_envs, 3), device=self.device)
        actions[:, 0] = np.clip(2.0 * (cmd[0] / self.P_MAX) - 1.0, -1.0, 1.0)
        actions[:, 1] = np.clip(2.0 * (cmd[1] / self.P_MAX) - 1.0, -1.0, 1.0)
        actions[:, 2] = np.clip(2.0 * (cmd[2] / self.P_MAX) - 1.0, -1.0, 1.0)
        
        self.t += self.dt
        return actions

# -----------------------------------------------------------------------------
# 5. Main Loop
# -----------------------------------------------------------------------------
def main():
    env = None
    sim_logs = []

    try:
        # Config Selection
        cfg_map = {
            ("A", False): PorcaroRLEnvCfg_ModelA, ("A", True): PorcaroRLEnvCfg_ModelA_DR,
            ("B", False): PorcaroRLEnvCfg_ModelB, ("B", True): PorcaroRLEnvCfg_ModelB_DR,
            ("C", False): PorcaroRLEnvCfg_ModelC, ("C", True): PorcaroRLEnvCfg_ModelC_DR,
        }
        target_cfg = cfg_map.get((args.model_type, args.dr))
        
        if target_cfg:
            print(f"\n[System] Config: {target_cfg.__name__}")
            env_cfg = target_cfg()
        else:
            print(f"\n[Warning] No specific config found. Using Base.")
            env_cfg = PorcaroRLEnvCfg()

        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1000.0
        
        # Remove Drum
        if args.no_drum:
            try:
                if hasattr(env_cfg, "drum_cfg"): env_cfg.drum_cfg.init_state.pos = (0.0, 0.0, -10.0)
                elif hasattr(env_cfg.scene, "drum"): env_cfg.scene.drum.init_state.pos = (0.0, 0.0, -10.0)
            except: pass
        
        env = PorcaroRLEnv(cfg=env_cfg)
        
        # Fixed dt for Validation (50Hz)
        dt_step = 0.02 
        print(f"[System] Simulation dt fixed to: {dt_step}s (50Hz)")

        agent = SignalGeneratorAgent(env.num_envs, dt_step, env.device, args)
        
        obs, _ = env.reset()
        print("[System] Simulation Started...")

        while simulation_app.is_running():
            actions = agent.get_action(obs)
            if agent.is_replay and agent.replay_finished:
                print("[System] Replay finished.")
                break

            # 変更点: extras を受け取るように修正
            obs, rew, terminated, truncated, extras = env.step(actions)
            
            # 変更点: env.py で計算済みのピーク力を extras から取得
            # extras["force/max_force_pooled"] には環境ごとの平均（ここでは1環境なのでその値）が入っています
            sim_force_n = extras.get("force/max_force_pooled", 0.0)
            if torch.is_tensor(sim_force_n):
                sim_force_n = sim_force_n.item()

            # 変更点: 観測値のインデックスを修正
            # env.py の _get_observations に基づき、0:手首角度, 1:グリップ角度, 2:手首速度, 3:グリップ速度
            policy_obs = obs["policy"][0]
            angle_wrist_deg = math.degrees(policy_obs[0].item())
            angle_grip_deg = math.degrees(policy_obs[1].item())
            vel_wrist_deg = math.degrees(policy_obs[2].item())
            
            sim_logs.append({
                'time': agent.t,
                'cmd_DF': agent.last_cmd[0],
                'cmd_F': agent.last_cmd[1],
                'cmd_G': agent.last_cmd[2],
                'sim_pres_DF': agent.pres_state[0],
                'sim_pres_F': agent.pres_state[1],
                'sim_pres_G': agent.pres_state[2],
                'sim_angle_deg': angle_wrist_deg,      # 手首角度
                'sim_angle_grip_deg': angle_grip_deg, # グリップ角度
                'sim_vel_deg': vel_wrist_deg,          # 手首速度
                'sim_force_n': sim_force_n,           # 変更点: 打撃力(N)を追加
                'model_type': args.model_type
            })

    except KeyboardInterrupt:
        print("\n[Info] Interrupted.")
    except Exception as e:
        print(f"\n[Error] {e}")
        traceback.print_exc()
    finally:
        # Save CSV
        if sim_logs:
            input_name = args.csv_name if args.csv_name else "manual"
            input_base = os.path.splitext(os.path.basename(input_name))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            filename = f"sim_log_Model{args.model_type}_{input_base}_{timestamp}.csv"
            save_dir = os.path.join(os.getcwd(), "verification_logs")
            os.makedirs(save_dir, exist_ok=True)
            
            path = os.path.join(save_dir, filename)
            pd.DataFrame(sim_logs).to_csv(path, index=False)
            print(f"\n[Saved] Sim Log: {path}")
            print(f"  -> Contains: time, cmd_*, sim_pres_*, sim_angle_deg, sim_vel_deg")
        
        if env: env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()