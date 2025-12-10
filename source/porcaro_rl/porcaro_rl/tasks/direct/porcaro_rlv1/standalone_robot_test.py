import argparse
import torch
import math
import numpy as np
import time
import datetime
from pathlib import Path
import csv, os, atexit, signal, sys, tempfile




from isaaclab.app import AppLauncher


# App起動後にモジュールをインポート
app_launcher = AppLauncher()
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.terrains import TerrainImporter
from isaaclab.sensors import ContactSensor
from isaaclab.sim import schemas as sim_schemas
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.utils.timer import Timer

from porcaro_rl_env_cfg import (
    PorcaroRLEnvCfg, 
    ROBOT_CFG, DRUM_CFG,
    WRIST_J0, GRIP_J0,
    contact_forces_stick_at_drum_cfg,
    drum_vs_stick_cfg,
    ASSETS_DIR,
    FORCE_MAP_CSV, H0_MAP_CSV,
)

from actions.torque import TorqueActionController  # actions/torque.py

# ===== パラメータ（必要最小限） =====
CTRL_HZ = 120.0            # トルク演算の更新周期 [Hz]
FREQ_HZ = 2.0              # サイン波の周波数 [Hz]
PRINT_EVERY = 20           # 何ステップごとに表示するか

# PAM-DF, PAM-F, PAM-G の幾何・応答（旧RLと同等）
GEOM_R = 0.014             # プーリ半径 r [m]
GEOM_L = 0.150             # 筋長 L [m]
THETA_T_DF = 0.0           # [deg]
THETA_T_F  = -90.0         # [deg]
THETA_T_G  = -45.0         # [deg]
P_MAX = 0.6                # [MPa] 入力圧上限（スケーリング）
FORCE_SCALE_N = 630        # 近似スケール（マップがあればそちら優先）


# ==== パラメータ（必要に応じて変更） ====
FREQ_HZ = 2.0                 # サイン波の周波数[Hz]
AMP_DEG = 25.0                # サイン波の振幅[deg]
PRINT_EVERY = 20              # 何ステップごとに出力するか

PAM_DF_SEQ = [0.5, 0.5, 0.0, 0.0, 0.0]  # [MPa]
PAM_F_SEQ  = [0.0, 0.0, 0.5, 0.5, 0.0]  # [MPa]
PAM_G_SEQ  = [0.0, 0.0, 0.5, 0.0, 0.0]  # [MPa]
STEP_DURATION = 0.2  # [s]

# 追加: 引数
parser = argparse.ArgumentParser()
parser.add_argument("--print-every", type=int, default=PRINT_EVERY)
parser.add_argument("--amp-deg", type=float, default=AMP_DEG)
parser.add_argument("--ctrl-hz", type=float, default=CTRL_HZ)
parser.add_argument("--log-dir", type=str, default=".")
parser.add_argument("--basename", type=str, default="force_log")
parser.add_argument("--no-render", action="store_true")
args, _ = parser.parse_known_args()

PRINT_EVERY = max(1, args.print_every)
AMP_DEG = args.amp_deg
CTRL_HZ = args.ctrl_hz
LOG_DIR = Path(args.log_dir)
LOG_DIR.mkdir(parents=True, exist_ok=True)
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH = LOG_DIR / f"{args.basename}_{STAMP}.csv"

class SafeCSVLogger:
    def __init__(self, path: Path, header: list[str]):
        self.path = Path(path)
        self.tmp_path = self.path.with_suffix(self.path.suffix + ".part")
        self.header = header
        self.file = open(self.tmp_path, "w", newline="", buffering=1)  # 行バッファ
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.header)
        self.rows_written = 0

    def write_row(self, row):
        self.writer.writerow(row)
        self.rows_written += 1
        # こまめにフラッシュ + fsync（クラッシュ対策）
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        try:
            if not self.file.closed:
                self.file.flush()
                os.fsync(self.file.fileno())
                self.file.close()
            # アトミックに .part -> 本ファイルへ
            os.replace(self.tmp_path, self.path)
        except Exception as e:
            print(f"[ERR] CSV finalize failed: {e}", flush=True)

def _install_signal_handlers(finalizer):
    def _handler(signum, frame):
        print(f"\n[WARN] Caught signal {signum}. Finalizing CSV...", flush=True)
        try:
            finalizer()
        finally:
            # 同じシグナルを既定動作に戻して再送（適切に終了）
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass



def main():
    force_log = []

    # ... 既存の設定生成の前後どこでもOK（早め推奨） ...
    header = ["time(s)","f_net_mag(N)","f_drum_mag(N)","f_drum_x(N)","f_drum_y(N)","f_drum_z(N)"]
    csvlogger = SafeCSVLogger(CSV_PATH, header)

    # atexit と signal で“ほぼ必ず”CSVを確定
    def _finalize():
        csvlogger.close()
    atexit.register(_finalize)
    _install_signal_handlers(_finalize)

    """Isaac Lab 環境でアセットをスポーンして静的に表示する"""
    # 1) 環境設定を読み込み
    cfg = PorcaroRLEnvCfg()

    # 2) シミュレーションを構築
    sim = sim_utils.SimulationContext(cfg.sim)

    # 3) シーン（地面と照明）を生成
    TerrainImporter(cfg.terrain)
    prim_path = "/World/defaultLight"
    light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func(prim_path, light_cfg)

    # 4) アセットのパスを上書き
    ROBOT_CFG.prim_path = "/World/Robot"
    DRUM_CFG.prim_path = "/World/Drum"

    # 5) アセットを生成
    robot = Articulation(ROBOT_CFG)
    drum = RigidObject(DRUM_CFG)

    sim_schemas.activate_contact_sensors("/World/Robot", threshold=0.0)
    sim_schemas.activate_contact_sensors("/World/Drum",  threshold=0.0)

    stick_vs_drum =ContactSensor(contact_forces_stick_at_drum_cfg)
    drum_vs_stick = ContactSensor(drum_vs_stick_cfg)

    dt_sim  = float(cfg.sim.dt)
    dt_ctrl = 1.0 / CTRL_HZ if CTRL_HZ > 0 else dt_sim
    ctrl = TorqueActionController(
        dt_ctrl=dt_ctrl,
        r=GEOM_R, L=GEOM_L,
        theta_t_DF_deg=THETA_T_DF,
        theta_t_F_deg=THETA_T_F,
        theta_t_G_deg=THETA_T_G,
        Pmax=P_MAX, N=FORCE_SCALE_N,
        force_map_csv=FORCE_MAP_CSV,  # assets/map.csv を使う
        h0_map_csv=H0_MAP_CSV,
    )

    
    # 6) シミュレーションを開始
    sim.reset()
    stick_vs_drum.reset(); drum_vs_stick.reset()

    ctrl.reset(n_envs=1, device=robot.data.joint_pos.device)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    # 関節名の順序に合わせて該当indexを書き換え
    names = list(robot.data.joint_names)  # すべての関節名（トルク対象関節の順）
    def idx(n): return names.index(n)

    joint_pos[:, idx("Base_link_Wrist_joint")] = WRIST_J0
    joint_pos[:, idx("Hand_link_Grip_joint")]  = GRIP_J0

    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()  # 初期状態を確定（公式の流れ）


    for _ in range(3):
        sim.step(render=False)
    sim.play()

    # サイン波駆動用
    dt = cfg.sim.dt
    t  = 0.0
    amp = math.radians(AMP_DEG)
    step = 0

    # ループ開始の直前の案内を少しだけ調整
    print("\n===== Running. Contact force [N] is being RECORDED EVERY STEP. =====", flush=True)
    print(f"===== Console prints every {PRINT_EVERY} steps (to reduce overhead). =====", flush=True)
    print(f"===== CSV will be saved to: {CSV_PATH} =====\n", flush=True)


    # (A) try: は while ループの外側に配置
    try:
        # (B) while ループ開始
        while simulation_app.is_running():
            simulation_app.update()
            if not sim.is_playing():
                continue

            # --- サイン波を「圧力差→トルク」にして印加 ---
            t += dt

            idx_base = int(t // STEP_DURATION)

            # シーケンスを繰り返し取得するヘルパー関数
            def pick_repeating(seq, i):
                # シーケンスが空の場合は0を返す
                if not seq:
                    return 0.0
                # 剰余(%)を使ってインデックスをループさせる
                return seq[i % len(seq)]
            P_DF = pick_repeating(PAM_DF_SEQ, idx_base)
            P_F = pick_repeating(PAM_F_SEQ, idx_base)
            P_G = pick_repeating(PAM_G_SEQ, idx_base)

            # 目標圧力 [MPa] → actions へ逆変換
            # ... (P_to_action, a_DF, a_F, a_G の計算はそのまま) ...
            def P_to_action(P):
                P = max(0.0, min(P, P_MAX))
                return 2.0*(P / P_MAX) - 1.0

            a_DF = P_to_action(P_DF)
            a_F  = P_to_action(P_F)
            a_G  = P_to_action(P_G)
            
            actions = torch.tensor([[a_DF, a_F, a_G]], dtype=torch.float32, device=robot.data.joint_pos.device)
            ctrl.apply(
                actions=actions,
                q=robot.data.joint_pos,
                joint_ids=(idx("Base_link_Wrist_joint"), idx("Hand_link_Grip_joint")),
                robot=robot
            )

            robot.write_data_to_sim()

            # (C) 以下の処理は *すべて* while ループの *内側* に置く
            
            # --- 1ステップ進める ---
            sim.step()
            stick_vs_drum.update(dt)
            drum_vs_stick.update(dt)
            
            # --- センサーデータを毎ステップ取得 ---
            f_net = stick_vs_drum.data.net_forces_w
            f_mat = stick_vs_drum.data.force_matrix_w

            # --- データを計算 ---
            f_net_mag = float(torch.linalg.norm(f_net[0, 0])) if (f_net is not None and f_net.shape[2] > 0) else 0.0

            
            f_drum_mag = 0.0
            f_drum_vec = [0.0, 0.0, 0.0]
            if (f_mat is not None) and (f_mat.shape[2] > 0):
                # 最初の接触点のベクトル（ワールド座標）
                f_vec_tensor = f_mat[0, 0, 0]
                f_drum_mag = float(torch.linalg.norm(f_vec_tensor))
                f_drum_vec = f_vec_tensor.tolist()

            # --- 毎ステップ記録（重要: printとは分離） ---
            row = [t, f_net_mag, f_drum_mag] + f_drum_vec
            force_log.append(row)
            csvlogger.write_row(row)   # ← 逐次追記（ここが重要）

            # --- 画面への表示は間引き（軽量化）
            if step % PRINT_EVERY == 0:
                msg = (
                    f"t={t:6.3f}s | "
                    f"F_net(stick)= {f_net_mag:8.2f} N | "
                    f"F_drum= {(f'{f_drum_mag:8.2f} N' if f_drum_mag>0 else '(no match)')}"
                )
                print(msg, flush=True)
            
            step += 1
        
        # (D) while ループが終了したら、ここに来る

    # (E) except と finally は while ループの *外側* に配置
    except KeyboardInterrupt:
        print("Simulation stopped by user (Ctrl+C).")

    finally:
        print(f"\n===== Simulation ended. Logged {len(force_log)} rows. Finalizing CSV... =====", flush=True)
        try:
            # atexit 側でも close するが、二重 close 安全
            csvlogger.close()
            print(f"Saved to '{CSV_PATH.name}' in '{LOG_DIR.resolve()}'", flush=True)
        except Exception as e:
            print(f"[ERR] finalize failed: {e}", flush=True)

        time.sleep(0.2)
        simulation_app.close()
    # (↑ main() 関数はここまで)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc()