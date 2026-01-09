"""
Porcaro Robot: Manual Verification Script for Sim-to-Real
---------------------------------------------------------
強化学習エージェントの代わりに、手動で定義した信号（サイン波、ステップ波など）を
ロボットに入力し、挙動を検証・ログ取得するためのスクリプトです。

Usage:
    python scripts/verification/run_manual_verification.py --mode sine --freq 1.0
    python scripts/verification/run_manual_verification.py --mode step --step_duration 2.0
    python scripts/verification/run_manual_verification.py --mode const --const_val 0.3

Modes:
    sine  : サイン波 (拮抗駆動または単一駆動)
    step  : 0.1MPa刻みの階段状入力 (0.0 -> 0.1 -> ... -> 0.6 -> 0.0 ...)
    const : 一定圧力
"""

import argparse
import torch
import numpy as np
import math
from isaaclab.app import AppLauncher

# 1. 引数定義 (App起動前に設定)
parser = argparse.ArgumentParser(description="Manual Control Verification")

# 検証モード設定
parser.add_argument("--mode", type=str, default="sine", choices=["sine", "step", "const"], help="Input signal mode")
parser.add_argument("--target", type=str, default="wrist_antagonistic", 
                    choices=["wrist_antagonistic", "wrist_df_only", "grip_only", "all_same"],
                    help="Which muscle to drive")

# サイン波パラメータ
parser.add_argument("--freq", type=float, default=1.0, help="Frequency [Hz]")
parser.add_argument("--amp", type=float, default=0.25, help="Amplitude [MPa] (Peak-to-Peak/2)")
parser.add_argument("--offset", type=float, default=0.3, help="Offset pressure [MPa]")

# ステップ/定値パラメータ
parser.add_argument("--step_duration", type=float, default=1.0, help="Duration per step level [s]")
parser.add_argument("--const_val", type=float, default=0.3, help="Pressure for const mode [MPa]")

# 環境設定
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--headless", action="store_true", help="Run without GUI")

# Isaac Sim 引数の処理
args, unknown = parser.parse_known_args()

# 2. App起動
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# 3. 環境のインポート (App起動後)
from porcaro_rl.tasks.direct.porcaro_rlv1.porcaro_rl_env import PorcaroRLEnv
from porcaro_rl.tasks.direct.porcaro_rlv1.porcaro_rl_env_cfg import PorcaroRLEnvCfg

class SignalGeneratorAgent:
    """
    指定されたパターンで圧力を計算し、Env用のアクション[-1, 1]を出力するエージェント
    """
    def __init__(self, num_envs, dt, device, args):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.args = args
        self.t = 0.0
        
        # 環境の最大圧力定義 (Env/Controller設定と一致させること)
        self.P_MAX = 0.6 
        
        # ステップ入力用のレベル定義 (0.0 ~ 0.6 MPa)
        self.step_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        print(f"\n[Agent] Initialized in mode: {args.mode}")
        print(f"[Agent] Target muscles: {args.target}")
        if args.mode == "sine":
            print(f"[Agent] Sine Wave: Freq={args.freq}Hz, Amp={args.amp}MPa, Offset={args.offset}MPa")
        elif args.mode == "step":
            print(f"[Agent] Step Sequence: {self.step_levels} MPa (Duration={args.step_duration}s)")
        
    def _pressure_to_action(self, p_mpa):
        """圧力[MPa] を Action[-1, 1] に変換"""
        # クリップ
        p = np.clip(p_mpa, 0.0, self.P_MAX)
        # 変換: 0 -> -1, P_MAX -> 1
        return 2.0 * (p / self.P_MAX) - 1.0

    def get_action(self, obs=None):
        # 1. 基準となる信号圧力を計算
        base_signal = 0.0
        
        if self.args.mode == "sine":
            # P = Offset + A * sin(2πft)
            base_signal = self.args.offset + self.args.amp * math.sin(2 * math.pi * self.args.freq * self.t)
            
        elif self.args.mode == "const":
            base_signal = self.args.const_val
            
        elif self.args.mode == "step":
            # 時間経過でインデックスを進める
            idx = int(self.t / self.args.step_duration) % len(self.step_levels)
            base_signal = self.step_levels[idx]

        # 2. 筋肉ごとの配分 (Targetに応じて振り分け)
        # Action順序: [0]:Wrist_DF, [1]:Wrist_F, [2]:Grip (環境の仕様依存)
        
        p_df = 0.0 # 背屈 (持ち上げ)
        p_f  = 0.0 # 掌屈 (振り下ろし)
        p_g  = 0.0 # 把持
        
        if self.args.target == "wrist_antagonistic":
            # 拮抗駆動 (サイン波のときのみ逆位相、それ以外は同相または片側)
            if self.args.mode == "sine":
                # Offsetを中心に逆位相で振る
                p_df = base_signal
                # F側は逆位相: Offset - (Signal - Offset)
                p_f  = self.args.offset - (base_signal - self.args.offset)
            else:
                # ステップ等の場合は片側だけ動かすのが一般的だが、ここではDFに入れる
                p_df = base_signal
                p_f  = 0.0 # もしくは定圧
            p_g = 0.3 # グリップは適当に固定

        elif self.args.target == "wrist_df_only":
            p_df = base_signal
            p_f  = 0.0
            p_g  = 0.3

        elif self.args.target == "grip_only":
            p_df = 0.0 # Wristは脱力 (あるいは固定)
            p_f  = 0.0
            p_g  = base_signal
            
        elif self.args.target == "all_same":
            p_df = base_signal
            p_f  = base_signal
            p_g  = base_signal

        # 3. アクション変換
        act_df = self._pressure_to_action(p_df)
        act_f  = self._pressure_to_action(p_f)
        act_g  = self._pressure_to_action(p_g)
        
        # 4. テンソル生成
        actions = torch.zeros((self.num_envs, 3), device=self.device)
        actions[:, 0] = act_df
        actions[:, 1] = act_f
        actions[:, 2] = act_g
        
        # 時間更新
        self.t += self.dt
        
        return actions

def main():
    # コンフィグ設定
    env_cfg = PorcaroRLEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    
    # ロギングを強制有効化 (検証なので必ず記録)
    env_cfg.logging.enabled = True
    # ファイル名にモードを含める
    log_name = f"log_verification_{args.mode}_{args.target}.csv"
    env_cfg.logging.filepath = log_name
    
    print(f"\n[Info] Simulation Start. Log will be saved to: {log_name}")

    # 環境生成
    env = PorcaroRLEnv(cfg=env_cfg)
    
    # 制御周期 (RLステップ時間)
    dt_step = env.cfg.sim.dt * env.cfg.decimation # 0.005 * 4 = 0.02s
    
    # Agent生成
    agent = SignalGeneratorAgent(
        num_envs=env.num_envs,
        dt=dt_step,
        device=env.device,
        args=args
    )

    # 実行ループ
    obs, _ = env.reset()
    
    while simulation_app.is_running():
        # 1. Action取得 (Agent)
        actions = agent.get_action(obs)
        
        # 2. Physics Step & Logging (Env)
        obs, rew, terminated, truncated, info = env.step(actions)
        
        # 3. リセット処理
        if terminated.any() or truncated.any():
            env.reset_done(terminated | truncated)
            # 連続的な波形を見たい場合は時刻をリセットしない等の工夫も可だが、
            # ここではシンプルに継続する (Agent内の self.t はリセットされない)

    # 終了処理
    print("[Info] Closing environment...")
    env.close()

if __name__ == "__main__":
    main()