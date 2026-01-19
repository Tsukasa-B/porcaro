"""
Porcaro Robot: Sim-to-Real Gap Analysis Script (Full State Verification)
Usage:
    python scripts/analyze_sim_real.py
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def main():
    # ---------------------------------------------------------
    # 1. パスの自動解決
    # ---------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 実機データとSimデータのファイル名
    # ※ 必要に応じてファイル名を変更してください
    real_csv_name = "01-11-hysteresis.csv"
    sim_csv_name = "verify_ModelA_Ideal_REPLAY_01-11-hysteresis.csv"

    real_csv_path = os.path.join(project_root, "external_data", "jetson_project", real_csv_name)
    sim_csv_path = os.path.join(project_root, sim_csv_name)

    print(f"[Info] Loading Real Data: {real_csv_path}")
    print(f"[Info] Loading Sim Data : {sim_csv_path}")

    # ---------------------------------------------------------
    # 2. ファイル読み込み
    # ---------------------------------------------------------
    if not os.path.exists(real_csv_path):
        print(f"[Error] Real data file not found: {real_csv_path}")
        return
    if not os.path.exists(sim_csv_path):
        print(f"[Error] Sim data file not found: {sim_csv_path}")
        return

    real_df = pd.read_csv(real_csv_path)
    sim_df = pd.read_csv(sim_csv_path)

    # ---------------------------------------------------------
    # 3. データ処理 & 同期
    # ---------------------------------------------------------
    # 実機データ: 開始時刻を0にする
    t_real = real_df['timestamp'].values
    t_real = t_real - t_real[0]

    # Simデータ: time_s をそのまま使用
    t_sim = sim_df['time_s'].values

    # ---------------------------------------------------------
    # 4. 定量評価 (RMSE計算)
    # ---------------------------------------------------------
    # Simデータを実機時間軸にリサンプリングして誤差を計算

    # (A) Wrist Angle
    f_wrist = interp1d(t_sim, sim_df['q_wrist_deg'], kind='linear', fill_value="extrapolate")
    sim_wrist_resampled = f_wrist(t_real)
    rmse_wrist = np.sqrt(np.mean((real_df['meas_ang_wrist'] - sim_wrist_resampled)**2))

    # (B) Hand/Grip Angle
    if 'q_grip_deg' in sim_df.columns:
        f_hand = interp1d(t_sim, sim_df['q_grip_deg'], kind='linear', fill_value="extrapolate")
        sim_hand_resampled = f_hand(t_real)
        rmse_hand = np.sqrt(np.mean((real_df['meas_ang_hand'] - sim_hand_resampled)**2))
    else:
        rmse_hand = 0.0
        print("[Warn] 'q_grip_deg' column not found in Sim data.")

    print("="*50)
    print(f" Sim-to-Real Analysis Result")
    print("="*50)
    print(f" RMSE (Wrist): {rmse_wrist:.4f} deg")
    print(f" RMSE (Hand) : {rmse_hand:.4f} deg")
    print("="*50)

    # ---------------------------------------------------------
    # 5. プロット (5行構成)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
    
    # 共通のスタイル設定関数
    def setup_ax(axis, ylabel, title=None):
        axis.set_ylabel(ylabel, fontsize=11)
        if title:
            axis.set_title(title, fontsize=13, fontweight='bold')
        axis.grid(True, linestyle='--', alpha=0.6)
        axis.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # --- Row 1: Wrist Angle ---
    ax[0].plot(t_real, real_df['meas_ang_wrist'], 'k-', label='Real Wrist', linewidth=2, alpha=0.8)
    ax[0].plot(t_sim, sim_df['q_wrist_deg'], 'r--', label='Sim Wrist', linewidth=2, alpha=0.8)
    setup_ax(ax[0], 'Angle [deg]', f'Wrist Angle (RMSE: {rmse_wrist:.2f} deg)')

    # --- Row 2: Hand (Grip) Angle ---
    ax[1].plot(t_real, real_df['meas_ang_hand'], 'k-', label='Real Hand', linewidth=2, alpha=0.8)
    if 'q_grip_deg' in sim_df.columns:
        ax[1].plot(t_sim, sim_df['q_grip_deg'], 'r--', label='Sim Hand', linewidth=2, alpha=0.8)
    setup_ax(ax[1], 'Angle [deg]', f'Hand/Grip Angle (RMSE: {rmse_hand:.2f} deg)')

    # --- Row 3: Pressure DF (Dorsi-Flexion) ---
    ax[2].plot(t_real, real_df['cmd_DF'], 'g:', label='Cmd DF', linewidth=1.5)
    ax[2].plot(t_real, real_df['meas_pres_DF'], 'k-', label='Real Meas DF', linewidth=1.5)
    if 'P_out_DF' in sim_df.columns:
        ax[2].plot(t_sim, sim_df['P_out_DF'], 'r--', label='Sim Out DF', linewidth=1.5)
    setup_ax(ax[2], 'Pressure [MPa]', 'PAM DF (Dorsi-Flexion)')

    # --- Row 4: Pressure F (Flexion) ---
    ax[3].plot(t_real, real_df['cmd_F'], 'g:', label='Cmd F', linewidth=1.5)
    ax[3].plot(t_real, real_df['meas_pres_F'], 'k-', label='Real Meas F', linewidth=1.5)
    if 'P_out_F' in sim_df.columns:
        ax[3].plot(t_sim, sim_df['P_out_F'], 'r--', label='Sim Out F', linewidth=1.5)
    setup_ax(ax[3], 'Pressure [MPa]', 'PAM F (Flexion)')

    # --- Row 5: Pressure G (Grip) ---
    ax[4].plot(t_real, real_df['cmd_G'], 'g:', label='Cmd G', linewidth=1.5)
    ax[4].plot(t_real, real_df['meas_pres_G'], 'k-', label='Real Meas G', linewidth=1.5)
    if 'P_out_G' in sim_df.columns:
        ax[4].plot(t_sim, sim_df['P_out_G'], 'r--', label='Sim Out G', linewidth=1.5)
    setup_ax(ax[4], 'Pressure [MPa]', 'PAM G (Grip)')
    
    ax[4].set_xlabel('Time [s]', fontsize=12)

    plt.tight_layout()
    
    # 画像保存
    output_img = os.path.join(project_root, "sim_real_comparison_full.png")
    plt.savefig(output_img)
    print(f"[Output] Graph saved to: {output_img}")
    
    # GUI表示 (可能な場合)
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    main()