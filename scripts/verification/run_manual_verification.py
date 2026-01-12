"""
Porcaro Robot: Multi-Model Manual Verification Script (Sim-to-Real Ready)
"""
import argparse
import sys
import traceback
import torch
import numpy as np
import math

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# 1. 引数定義
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Multi-Model Verification Script")

# --- 実験モード設定 ---
parser.add_argument("--model_type", type=str, default="A", choices=["A", "B", "C"], 
                    help="Model Type: A (Ideal/Delay), B (Hysteresis), C (ActuatorNet)")
parser.add_argument("--dr", action="store_true", help="Enable Domain Randomization (DR)")

# --- 入力信号設定 ---
parser.add_argument("--mode", type=str, default="double", choices=["sine", "step", "const", "double"], 
                    help="Input signal pattern")
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
parser.add_argument("--video", action="store_true", help="Record video (requires headless)")

args, unknown = parser.parse_known_args()

# -----------------------------------------------------------------------------
# 2. App起動 & インポート
# -----------------------------------------------------------------------------
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

try:
    # モデルごとのConfigクラスを動的にインポート
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
# 3. エージェントクラス (信号生成)
# -----------------------------------------------------------------------------
class SignalGeneratorAgent:
    def __init__(self, num_envs, dt, device, args):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.args = args
        self.t = 0.0
        self.P_MAX = 0.6 
        
        # リズムパターンの解析
        self.pattern_seq = [int(x) for x in args.pattern.split(",")]
        self.beat_sec = 60.0 / args.bpm
        self.note_sec = self.beat_sec / 4.0 # 16分音符基準
        self.total_cycle_sec = self.note_sec * len(self.pattern_seq)

        print(f"\n[Agent] Mode: {args.mode}")
        if args.mode == "double":
            print(f" -> BPM: {args.bpm}, Pattern: {self.pattern_seq}")
            print(f" -> High P: {args.pressure_high}, Low P: {args.pressure_low}")

    def _pressure_to_action(self, p_mpa):
        """0~0.6MPa を -1~1 に変換"""
        p = np.clip(p_mpa, 0.0, self.P_MAX)
        return 2.0 * (p / self.P_MAX) - 1.0

    def get_action(self, obs=None):
        p_df, p_f, p_g = 0.0, 0.0, 0.0
        
        # --- Rhythm Pattern Mode ---
        if self.args.mode == "double":
            t_in_cycle = self.t % self.total_cycle_sec
            note_idx = int(t_in_cycle / self.note_sec)
            t_in_note = t_in_cycle % self.note_sec
            
            # 安全のため範囲チェック
            if note_idx >= len(self.pattern_seq): note_idx = 0
            is_hit = self.pattern_seq[note_idx] == 1
            
            # デフォルト: 待機 (F=Low, DF=High)
            target_f = self.args.pressure_low
            target_df = self.args.pressure_high
            
            if is_hit:
                # 叩く動作: (F=High, DF=Low)
                if t_in_note < (self.note_sec * self.args.duty_cycle):
                    target_f = self.args.pressure_high
                    target_df = self.args.pressure_low
            
            p_f = target_f
            p_df = target_df
            p_g = self.args.pressure_grip

        # --- Sine Wave Mode (Sweep Test) ---
        elif self.args.mode == "sine":
            # 拮抗駆動で逆位相のサイン波を入れる (ヒステリシス確認用)
            sine_val = math.sin(2 * math.pi * self.args.freq * self.t)
            base = self.args.offset + self.args.amp * sine_val
            
            p_df = base
            p_f = self.args.offset - (base - self.args.offset) # 逆位相
            p_g = self.args.pressure_grip
            
        # --- Step Response Mode ---
        elif self.args.mode == "step":
            # 一定時間ごとに High/Low を切り替える
            is_high = int(self.t / self.args.step_interval) % 2 == 0
            val = self.args.pressure_high if is_high else self.args.pressure_low
            
            p_df = self.args.pressure_low # DFは緩める
            p_f = val                     # Fをステップ駆動
            p_g = self.args.pressure_grip
            
        elif self.args.mode == "const":
            p_df = self.args.offset
            p_f = self.args.offset
            p_g = self.args.pressure_grip

        # Action生成
        actions = torch.zeros((self.num_envs, 3), device=self.device)
        actions[:, 0] = self._pressure_to_action(p_df)
        actions[:, 1] = self._pressure_to_action(p_f)
        actions[:, 2] = self._pressure_to_action(p_g)
        
        self.t += self.dt
        return actions

# -----------------------------------------------------------------------------
# 4. メイン処理
# -----------------------------------------------------------------------------
def main():
    env = None
    try:
        # --- Config選択ロジック ---
        # ユーザー指定のモデルタイプとDR有無に応じてクラスを選ぶ
        cfg_map = {
            ("A", False): PorcaroRLEnvCfg_ModelA,
            ("A", True):  PorcaroRLEnvCfg_ModelA_DR,
            ("B", False): PorcaroRLEnvCfg_ModelB,
            ("B", True):  PorcaroRLEnvCfg_ModelB_DR,
            ("C", False): PorcaroRLEnvCfg_ModelC,
            ("C", True):  PorcaroRLEnvCfg_ModelC_DR,
        }
        
        target_cfg_cls = cfg_map.get((args.model_type, args.dr))
        if target_cfg_cls is None:
            raise ValueError("Invalid Model/DR combination.")
            
        print(f"\n[System] Selecting Config: {target_cfg_cls.__name__}")
        
        # インスタンス化 & 設定上書き
        env_cfg = target_cfg_cls()
        env_cfg.scene.num_envs = args.num_envs
        # マニュアル検証なので直接圧力制御モードにしておくのが無難
        env_cfg.controller.control_mode = "pressure" 
        
        # ログファイル名の生成
        dr_str = "DR" if args.dr else "Ideal"
        log_name = f"verify_Model{args.model_type}_{dr_str}_{args.mode}_{int(args.bpm)}bpm.csv"
        
        if hasattr(env_cfg, "logging"):
            env_cfg.logging.enabled = True
            env_cfg.logging.filepath = log_name

        print(f"[System] Log file: {log_name}")

        # 環境起動
        env = PorcaroRLEnv(cfg=env_cfg)
        dt_step = env.cfg.sim.dt * env.cfg.decimation
        
        # エージェント作成
        agent = SignalGeneratorAgent(env.num_envs, dt_step, env.device, args)
        
        obs, _ = env.reset()
        print("[System] Simulation Loop Started...")

        while simulation_app.is_running():
            actions = agent.get_action(obs)
            
            # Step実行
            obs, rew, terminated, truncated, info = env.step(actions)
            
            # タイムアウト等でのリセットは環境内で自動処理されるが、
            # マニュアル検証用に強制リセットしたい場合はここで handle
            if (terminated | truncated).any():
                env.reset_idx(torch.where(terminated | truncated)[0])
                
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