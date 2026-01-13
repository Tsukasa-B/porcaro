"""
Porcaro Robot: Sim-to-Real Gap Analysis Script
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
    # 1. パスの自動解決 (スクリプトの場所を基準にする)
    # ---------------------------------------------------------
    # このスクリプトがある場所 (.../porcaro_rl/scripts)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # プロジェクトルート (.../porcaro_rl)
    project_root = os.path.dirname(script_dir)

    # A. 実機データのパス
    # 構成: porcaro_rl/external_data/jetson_project/01-11-step.csv
    real_csv_name = "01-11-step.csv"
    real_csv_path = os.path.join(project_root, "external_data", "jetson_project", real_csv_name)

    # B. Simログのパス
    # 構成: porcaro_rl/verify_ModelA_... (scriptsと同じ階層＝ルート直下)
    sim_csv_name = "verify_ModelA_Ideal_REPLAY_01-11-step.csv"
    sim_csv_path = os.path.join(project_root, sim_csv_name)

    print(f"[Info] Loading Real Data: {real_csv_path}")
    print(f"[Info] Loading Sim Data : {sim_csv_path}")

    # ---------------------------------------------------------
    # 2. ファイル読み込みチェック
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

    # 角度データの抽出 (実機: meas_ang_wrist, Sim: q_wrist_deg)
    # Sim側のデータを、実機の時間軸に合わせて補間(Resample)する
    f_ang = interp1d(t_sim, sim_df['q_wrist_deg'], kind='linear', fill_value="extrapolate")
    sim_ang_resampled = f_ang(t_real)

    # ---------------------------------------------------------
    # 4. 定量評価 (RMSE計算)
    # ---------------------------------------------------------
    # グラフで見やすいように、明らかにデータがない区間(開始直後など)を除くなどの処理が必要ならここで行う
    # 今回は全区間で計算
    rmse = np.sqrt(np.mean((real_df['meas_ang_wrist'] - sim_ang_resampled)**2))
    print("="*40)
    print(f" Sim-to-Real Analysis Result")
    print("="*40)
    print(f" Target File : {real_csv_name}")
    print(f" RMSE (Angle): {rmse:.4f} deg")
    print("="*40)

    # ---------------------------------------------------------
    # 5. プロット
    # ---------------------------------------------------------
    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # (1) Angle Comparison
    ax[0].plot(t_real, real_df['meas_ang_wrist'], 'k-', label='Real Robot', linewidth=2, alpha=0.8)
    ax[0].plot(t_sim, sim_df['q_wrist_deg'], 'r--', label='Sim (Model A)', linewidth=2, alpha=0.8)
    ax[0].set_ylabel('Wrist Angle [deg]', fontsize=12)
    ax[0].set_title(f'Sim-to-Real Gap Analysis (RMSE: {rmse:.2f} deg)', fontsize=14)
    ax[0].legend(loc='upper right')
    ax[0].grid(True, linestyle='--', alpha=0.7)

    # (2) Pressure Comparison (DF) - 背屈側
    ax[1].plot(t_real, real_df['cmd_DF'], 'g:', label='Command (DF)', linewidth=1.5)
    ax[1].plot(t_real, real_df['meas_pres_DF'], 'k-', label='Real P_DF', linewidth=1.5)
    if 'P_out_DF' in sim_df.columns:
        ax[1].plot(t_sim, sim_df['P_out_DF'], 'r--', label='Sim P_DF', linewidth=1.5)
    ax[1].set_ylabel('Pressure DF [MPa]', fontsize=12)
    ax[1].legend(loc='upper right')
    ax[1].grid(True, linestyle='--', alpha=0.7)

    # (3) Pressure Comparison (F) - 掌屈側
    ax[2].plot(t_real, real_df['cmd_F'], 'g:', label='Command (F)', linewidth=1.5)
    ax[2].plot(t_real, real_df['meas_pres_F'], 'k-', label='Real P_F', linewidth=1.5)
    if 'P_out_F' in sim_df.columns:
        ax[2].plot(t_sim, sim_df['P_out_F'], 'r--', label='Sim P_F', linewidth=1.5)
    ax[2].set_ylabel('Pressure F [MPa]', fontsize=12)
    ax[2].set_xlabel('Time [s]', fontsize=12)
    ax[2].legend(loc='upper right')
    ax[2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    # 画像として保存 (GUIがない環境でも確認できるように)
    output_img = os.path.join(project_root, "sim_real_comparison.png")
    plt.savefig(output_img)
    print(f"[Output] Graph saved to: {output_img}")
    
    # GUIが使えるなら表示
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    main()