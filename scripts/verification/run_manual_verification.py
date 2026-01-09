"""
Porcaro Robot: Manual Verification Script (Antagonistic Control Version)
"""
import argparse
import sys
import traceback
from isaaclab.app import AppLauncher

# 1. 引数定義
parser = argparse.ArgumentParser(description="Manual Control Verification")
parser.add_argument("--mode", type=str, default="sine", choices=["sine", "step", "const", "double"], help="Input signal mode")
parser.add_argument("--target", type=str, default="wrist_antagonistic", choices=["wrist_antagonistic", "wrist_df_only", "grip_only", "all_same"])
parser.add_argument("--no_drum", action="store_true", help="Disable drum physics")
# パラメータ設定 (ダブルストローク用)
parser.add_argument("--bpm", type=float, default=160.0)
parser.add_argument("--duty_cycle", type=float, default=0.5, help="Ratio of Hit duration within a note")
parser.add_argument("--pressure_high", type=float, default=0.55, help="High pressure for active muscle [MPa]")
parser.add_argument("--pressure_low", type=float, default=0.05, help="Low pressure for inactive muscle [MPa]")
parser.add_argument("--pressure_grip", type=float, default=0.3, help="Grip pressure [MPa]")

# その他 (Sine/Step用)
parser.add_argument("--freq", type=float, default=1.0)
parser.add_argument("--amp", type=float, default=0.25)
parser.add_argument("--offset", type=float, default=0.3)
parser.add_argument("--const_val", type=float, default=0.3)
parser.add_argument("--step_duration", type=float, default=1.0)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--headless", action="store_true")

args, unknown = parser.parse_known_args()

# 2. App起動
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# 3. 環境インポート
try:
    import torch
    import numpy as np
    import math
    from porcaro_rl.tasks.direct.porcaro_rlv1.porcaro_rl_env import PorcaroRLEnv
    from porcaro_rl.tasks.direct.porcaro_rlv1.porcaro_rl_env_cfg import PorcaroRLEnvCfg
except Exception as e:
    print(f"[CRITICAL ERROR] Import failed: {e}")
    simulation_app.close()
    sys.exit(1)

class SignalGeneratorAgent:
    def __init__(self, num_envs, dt, device, args):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.args = args
        self.t = 0.0
        self.P_MAX = 0.6 
        self.step_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        print(f"\n[Agent] Initialized in mode: {args.mode}")
        if args.mode == "double":
            self.beat_sec = 60.0 / args.bpm
            self.note_sec = self.beat_sec / 4.0 # 16分音符
            print(f" -> BPM: {args.bpm}")
            print(f" -> High P: {args.pressure_high} MPa, Low P: {args.pressure_low} MPa")
            print(f" -> Grip P: {args.pressure_grip} MPa")

    def _pressure_to_action(self, p_mpa):
        p = np.clip(p_mpa, 0.0, self.P_MAX)
        return 2.0 * (p / self.P_MAX) - 1.0

    def get_action(self, obs=None):
        p_df, p_f, p_g = 0.0, 0.0, 0.0
        
        # --- Double Stroke Mode (拮抗駆動対応版) ---
        if self.args.mode == "double":
            # 1サイクル = 16分音符 x 4つ (R, R, -, -)
            cycle_duration = self.note_sec * 4.0
            t_in_cycle = self.t % cycle_duration
            note_idx = int(t_in_cycle / self.note_sec) # 0, 1, 2, 3
            t_in_note = t_in_cycle % self.note_sec
            
            # デフォルトは「腕を上げて待機」状態 (F=Low, DF=High)
            target_f = self.args.pressure_low
            target_df = self.args.pressure_high
            
            # 1打目(idx=0) と 2打目(idx=1) の処理
            if note_idx in [0, 1]:
                # Duty Cycle内なら「叩く」(F=High, DF=Low)
                if t_in_note < (self.note_sec * self.args.duty_cycle):
                    target_f = self.args.pressure_high
                    target_df = self.args.pressure_low
                # Duty Cycleを過ぎたら「戻す」(F=Low, DF=High) -> デフォルトのまま
            
            p_f = target_f
            p_df = target_df
            p_g = self.args.pressure_grip

        # --- Sine Wave Mode ---
        elif self.args.mode == "sine":
            base = self.args.offset + self.args.amp * math.sin(2 * math.pi * self.args.freq * self.t)
            # 逆位相を作る
            p_df = base
            p_f = self.args.offset - (base - self.args.offset)
            p_g = self.args.pressure_grip
            
        # --- Step / Const Mode ---
        elif self.args.mode == "step":
            idx = int(self.t / self.args.step_duration) % len(self.step_levels)
            val = self.step_levels[idx]
            p_df = val
            p_f = 0.1 # 適当な低圧
            p_g = 0.3
            
        elif self.args.mode == "const":
            p_df = self.args.const_val
            p_f = self.args.const_val
            p_g = 0.3

        # Action変換
        actions = torch.zeros((self.num_envs, 3), device=self.device)
        actions[:, 0] = self._pressure_to_action(p_df)
        actions[:, 1] = self._pressure_to_action(p_f)
        actions[:, 2] = self._pressure_to_action(p_g)
        
        self.t += self.dt
        return actions

def main():
    env = None
    try:
        # Config設定
        env_cfg = PorcaroRLEnvCfg()
        env_cfg.controller.control_mode = "pressure"
        env_cfg.scene.num_envs = args.num_envs
        
        # ログ名に拮抗駆動(Antago)であることを明記
        drum_status = "NoDrum" if args.no_drum else "WithDrum"
        log_name = f"log_{args.mode}_{int(args.bpm)}bpm_Antago_{drum_status}.csv"
        
        if hasattr(env_cfg, "logging"):
            env_cfg.logging.enabled = True
            env_cfg.logging.filepath = log_name

        if args.no_drum:
            print(f"[Config] 🚫 DRUM DISABLED")
            if hasattr(env_cfg, "drum_cfg"):
                env_cfg.drum_cfg.init_state.pos = (0.0, 0.0, -10.0)

        print(f"\n[Info] Simulation Start. Log: {log_name}")

        env = PorcaroRLEnv(cfg=env_cfg)
        dt_step = env.cfg.sim.dt * env.cfg.decimation
        agent = SignalGeneratorAgent(env.num_envs, dt_step, env.device, args)
        
        obs, _ = env.reset()
        
        while simulation_app.is_running():
            actions = agent.get_action(obs)
            obs, rew, terminated, truncated, info = env.step(actions)
            # 自動リセット

    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        traceback.print_exc()
    finally:
        if env is not None:
            print("\n[System] Saving logs...")
            try:
                env.close()
                print("[System] Save Complete.")
            except Exception as e:
                print(f"[Error] Failed to save log: {e}")
        simulation_app.close()

if __name__ == "__main__":
    main()