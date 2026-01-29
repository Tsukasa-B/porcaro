"""
Porcaro Robot: IROS Validation Analysis (Real vs Model A vs Model B)
Usage:
    python scripts/analyze_sim_real.py
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# =========================================================
# 設定: 比較対象のファイル名
# =========================================================
REAL_CSV = "data_exp4_validation_inverted.csv"
SIM_A_CSV = "verify_ModelA_Ideal_REPLAY_data_exp4_validation_inverted.csv"
SIM_B_CSV = "verify_ModelB_Ideal_REPLAY_data_exp4_validation_inverted.csv"

# =========================================================
# ヘルパー関数
# =========================================================
def calc_rmse(t_real, y_real, t_sim, y_sim):
    """
    Simデータを実機時間軸にリサンプリングしてRMSEを計算する
    """
    if len(y_sim) == 0: return 0.0
    
    # 線形補間
    f = interp1d(t_sim, y_sim, kind='linear', fill_value="extrapolate")
    y_sim_resampled = f(t_real)
    
    # RMSE計算
    return np.sqrt(np.mean((y_real - y_sim_resampled)**2))

def load_csv(path):
    if not os.path.exists(path):
        print(f"[Warn] File not found: {path}")
        return None
    print(f"[Info] Loading: {path}")
    return pd.read_csv(path)

def main():
    # 1. パス解決
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # porcaro root

    # 外部データフォルダとプロジェクトルートを探査
    search_paths = [
        project_root,
        os.path.join(project_root, "external_data", "jetson_project"),
    ]
    
    def find_path(fname):
        for p in search_paths:
            full = os.path.join(p, fname)
            if os.path.exists(full): return full
        return os.path.join(project_root, fname) # fallback

    path_real = find_path(REAL_CSV)
    path_a = find_path(SIM_A_CSV)
    path_b = find_path(SIM_B_CSV)

    # 2. データ読み込み
    df_real = load_csv(path_real)
    df_a = load_csv(path_a)
    df_b = load_csv(path_b)

    if df_real is None:
        print("[Error] Real data is missing. Cannot compare.")
        return

    # 3. 時間軸の正規化
    # 実機: timestamp - initial
    t_real = df_real['timestamp'].values
    t_real = t_real - t_real[0]

    # Sim: time_s (Simがない場合は空配列)
    t_a = df_a['time_s'].values if df_a is not None else np.array([])
    t_b = df_b['time_s'].values if df_b is not None else np.array([])

    # 4. 誤差計算 (RMSE)
    metrics = {
        "Wrist": {"A": 0.0, "B": 0.0},
        "Hand":  {"A": 0.0, "B": 0.0},
        "P_DF":  {"A": 0.0, "B": 0.0},
        "P_F":   {"A": 0.0, "B": 0.0},
        "P_G":   {"A": 0.0, "B": 0.0},
    }

    # --- 角度誤差 ---
    if df_a is not None:
        metrics["Wrist"]["A"] = calc_rmse(t_real, df_real['meas_ang_wrist'], t_a, df_a['q_wrist_deg'])
        metrics["Hand"]["A"]  = calc_rmse(t_real, df_real['meas_ang_hand'],  t_a, df_a['q_grip_deg'])
    
    if df_b is not None:
        metrics["Wrist"]["B"] = calc_rmse(t_real, df_real['meas_ang_wrist'], t_b, df_b['q_wrist_deg'])
        metrics["Hand"]["B"]  = calc_rmse(t_real, df_real['meas_ang_hand'],  t_b, df_b['q_grip_deg'])

    # --- 圧力誤差 ---
    for key, real_col, sim_col in [("P_DF", "meas_pres_DF", "P_out_DF"), 
                                   ("P_F",  "meas_pres_F",  "P_out_F"), 
                                   ("P_G",  "meas_pres_G",  "P_out_G")]:
        if df_a is not None and sim_col in df_a.columns:
            metrics[key]["A"] = calc_rmse(t_real, df_real[real_col], t_a, df_a[sim_col])
        if df_b is not None and sim_col in df_b.columns:
            metrics[key]["B"] = calc_rmse(t_real, df_real[real_col], t_b, df_b[sim_col])

    # 結果表示
    print("\n" + "="*60)
    print(f" IROS Validation Analysis (RMSE Comparison)")
    print("="*60)
    print(f" {'Metric':<10} | {'Model A (Ideal)':<20} | {'Model B (Prop.)':<20}")
    print("-" * 60)
    for k, v in metrics.items():
        unit = "deg" if k in ["Wrist", "Hand"] else "MPa"
        print(f" {k:<10} | {v['A']:.4f} {unit:<15} | {v['B']:.4f} {unit}")
    print("="*60 + "\n")

    # 5. プロット
    fig, ax = plt.subplots(5, 1, figsize=(14, 20), sharex=True)

    # ★修正: metric_key を明示的に受け取るように変更
    def plot_row(idx, real_data, col_name_sim, title, ylabel, metric_key, ylim=None):
        # Real (Black)
        ax[idx].plot(t_real, real_data, 'k-', linewidth=2.5, label='Real', alpha=0.7)
        
        # Model A (Blue Dashed)
        if df_a is not None and col_name_sim in df_a.columns:
            rmse = metrics[metric_key]["A"]
            ax[idx].plot(t_a, df_a[col_name_sim], 'b--', linewidth=2, label=f'Model A (RMSE:{rmse:.2f})')
            
        # Model B (Red Dashed)
        if df_b is not None and col_name_sim in df_b.columns:
            rmse = metrics[metric_key]["B"]
            ax[idx].plot(t_b, df_b[col_name_sim], 'r--', linewidth=2, label=f'Model B (RMSE:{rmse:.2f})')

        ax[idx].set_ylabel(ylabel, fontsize=12)
        ax[idx].set_title(title, fontsize=14, fontweight='bold')
        ax[idx].grid(True, linestyle=':', alpha=0.6)
        ax[idx].legend(loc='upper right', fontsize=10)
        if ylim: ax[idx].set_ylim(ylim)

    # Row 1: Wrist Angle
    plot_row(0, df_real['meas_ang_wrist'], 'q_wrist_deg', 'Wrist Angle Tracking', 'Angle [deg]', metric_key="Wrist")
    
    # Row 2: Hand Angle
    plot_row(1, df_real['meas_ang_hand'], 'q_grip_deg', 'Hand/Grip Angle Tracking', 'Angle [deg]', metric_key="Hand")

    # Row 3: Pressure DF
    ax[2].plot(t_real, df_real['cmd_DF'], 'g:', linewidth=1.5, label='Command', alpha=0.8)
    plot_row(2, df_real['meas_pres_DF'], 'P_out_DF', 'P_DF (Dorsi-Flexion) Response', 'Pressure [MPa]', metric_key="P_DF", ylim=(-0.05, 0.7))

    # Row 4: Pressure F
    ax[3].plot(t_real, df_real['cmd_F'], 'g:', linewidth=1.5, label='Command', alpha=0.8)
    plot_row(3, df_real['meas_pres_F'], 'P_out_F', 'P_F (Flexion) Response', 'Pressure [MPa]', metric_key="P_F", ylim=(-0.05, 0.7))

    # Row 5: Pressure G
    ax[4].plot(t_real, df_real['cmd_G'], 'g:', linewidth=1.5, label='Command', alpha=0.8)
    plot_row(4, df_real['meas_pres_G'], 'P_out_G', 'P_G (Grip) Response', 'Pressure [MPa]', metric_key="P_G", ylim=(-0.05, 0.7))

    ax[4].set_xlabel('Time [s]', fontsize=14)
    plt.tight_layout()

    # 保存
    out_path = os.path.join(project_root, "iros_validation_comparison.png")
    plt.savefig(out_path)
    print(f"[Output] Plot saved to: {out_path}")
    
    # 表示
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    main()