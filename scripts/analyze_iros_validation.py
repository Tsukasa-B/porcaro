# scripts/analyze_iros_validation_v3.py
"""
Porcaro Robot: IROS 2026 Validation Analysis Script (3-Way Comparison)
Comparison: Physical System vs Model A vs Model B
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ==========================================
# 設定 (Configuration)
# ==========================================
REAL_LABEL = "Physical System"  # "Real Robot" の代わり
MODEL_A_LABEL = "Model A (Baseline)"
MODEL_B_LABEL = "Model B (Proposed)"

STICK_LENGTH = 0.35
MAX_PRESSURE = 0.6

# 論文用プロットスタイル
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['figure.dpi'] = 300

def load_data(path, label):
    """CSVを読み込み、カラム名を統一・単位変換する"""
    if not path or not os.path.exists(path):
        return None
    
    df = pd.read_csv(path)
    # print(f"[{label}] Raw columns: {list(df.columns)}") # デバッグ用
    
    # --- 1. カラム名の統一 ---
    rename_map = {
        'cmd_pressure_DF': 'cmd_DF', 'cmd_pressure_F': 'cmd_F',
        'obs_angle': 'angle_deg', 'obs_joint_pos': 'angle_deg', 'joint_pos': 'angle_deg',
        'q_wrist_deg': 'angle_deg', 'time_s': 'time', 'timestamp': 'time'
    }
    df = df.rename(columns=rename_map)
    
    # --- 2. Simデータの圧力復元 ---
    if 'meas_pres_DF' not in df.columns:
        if 'action_0' in df.columns:
            # Normalize: -1~1 => 0~0.6
            df['meas_pres_DF'] = (df['action_0'] + 1.0) / 2.0 * MAX_PRESSURE
        elif 'cmd_DF' in df.columns:
             df['meas_pres_DF'] = df['cmd_DF']

    # --- 3. 時間軸の同期 ---
    if 'time' in df.columns:
        df['time'] = df['time'] - df['time'].iloc[0]
    else:
        df['time'] = np.arange(len(df)) * 0.02

    return df

def get_real_data_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    candidates = [
        filename,
        os.path.join(project_root, "external_data", "jetson_project", "IROS", "test_signals", filename),
        os.path.join(project_root, "IROS", "test_signals", filename)
    ]
    for path in candidates:
        if os.path.exists(path): return path
    return filename

def calc_rmse(y_true, y_pred):
    """RMSEを計算 (長さが違う場合は短い方に合わせる)"""
    n = min(len(y_true), len(y_pred))
    return np.sqrt(mean_squared_error(y_true[:n], y_pred[:n]))

# ==========================================
# Plotting Functions (3-Way)
# ==========================================
def plot_exp1(df_real, df_sim_a, df_sim_b, out_dir):
    """Static Hysteresis"""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Real
    if df_real is not None:
        ax.plot(df_real['meas_pres_DF'], df_real['angle_deg'], 'k-', label=REAL_LABEL, alpha=0.8, linewidth=2.5)
    
    # Sim A
    if df_sim_a is not None:
        ax.plot(df_sim_a['meas_pres_DF'], df_sim_a['angle_deg'], 'b--', label=MODEL_A_LABEL, alpha=0.8)

    # Sim B
    if df_sim_b is not None:
        ax.plot(df_sim_b['meas_pres_DF'], df_sim_b['angle_deg'], 'r-.', label=MODEL_B_LABEL, alpha=0.8)
    
    ax.set_xlabel('Pressure DF [MPa]')
    ax.set_ylabel('Joint Angle [deg]')
    ax.set_title('Exp 1: Static Hysteresis Comparison')
    ax.legend()
    ax.grid(True, linestyle=':')
    plt.savefig(os.path.join(out_dir, "exp1_hysteresis_comparison.png"))
    print("Saved: exp1_hysteresis_comparison.png")

def plot_exp2(df_real, df_sim_a, df_sim_b, out_dir):
    """Step Response"""
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # Angle
    if df_real is not None:
        axes[0].plot(df_real['time'], df_real['angle_deg'], 'k-', label=REAL_LABEL)
    if df_sim_a is not None:
        axes[0].plot(df_sim_a['time'], df_sim_a['angle_deg'], 'b--', label=MODEL_A_LABEL)
    if df_sim_b is not None:
        axes[0].plot(df_sim_b['time'], df_sim_b['angle_deg'], 'r-.', label=MODEL_B_LABEL)
    
    axes[0].set_ylabel('Angle [deg]')
    axes[0].legend()
    axes[0].grid(True)
    
    # Pressure (Command & Real Meas)
    if df_real is not None:
        axes[1].plot(df_real['time'], df_real['cmd_DF'], 'g-', label='Command', alpha=0.5)
        axes[1].plot(df_real['time'], df_real['meas_pres_DF'], 'k:', label='Measured Pressure')
    
    axes[1].set_ylabel('Pressure [MPa]')
    axes[1].set_xlabel('Time [s]')
    axes[1].legend()
    axes[1].grid(True)
    plt.savefig(os.path.join(out_dir, "exp2_step_comparison.png"))
    print("Saved: exp2_step_comparison.png")

def plot_exp3(df_real, df_sim_a, df_sim_b, out_dir):
    """Frequency Sweep"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # 1. Trajectory
    ax = axes[0]
    real_vals = None
    
    if df_real is not None:
        ax.plot(df_real['time'], df_real['angle_deg'], 'k-', label=REAL_LABEL, alpha=0.7)
        real_vals = df_real['angle_deg'].values

    if df_sim_a is not None:
        rmse = calc_rmse(real_vals, df_sim_a['angle_deg'].values) if real_vals is not None else 0
        ax.plot(df_sim_a['time'], df_sim_a['angle_deg'], 'b--', label=f"{MODEL_A_LABEL} (RMSE={rmse:.1f})")

    if df_sim_b is not None:
        rmse = calc_rmse(real_vals, df_sim_b['angle_deg'].values) if real_vals is not None else 0
        ax.plot(df_sim_b['time'], df_sim_b['angle_deg'], 'r-.', label=f"{MODEL_B_LABEL} (RMSE={rmse:.1f})")
    
    ax.set_ylabel('Angle [deg]')
    ax.set_title('Exp 3: Frequency Sweep Response')
    ax.legend(loc='upper right')
    ax.grid(True)
    
    # 2. Error Plot
    ax = axes[1]
    t = df_real['time'].values if df_real is not None else df_sim_a['time'].values
    min_len = len(t)
    
    if df_real is not None and df_sim_a is not None:
        err_a = df_real['angle_deg'].values[:min_len] - df_sim_a['angle_deg'].values[:min_len]
        ax.plot(t[:len(err_a)], np.abs(err_a), 'b--', alpha=0.5, label='Error Model A')
        
    if df_real is not None and df_sim_b is not None:
        err_b = df_real['angle_deg'].values[:min_len] - df_sim_b['angle_deg'].values[:min_len]
        ax.plot(t[:len(err_b)], np.abs(err_b), 'r-.', alpha=0.5, label='Error Model B')

    ax.set_ylabel('Abs Error [deg]')
    ax.set_xlabel('Time [s]')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp3_sweep_comparison.png"))
    print("Saved: exp3_sweep_comparison.png")

def calculate_velocity_proxy(time, angle):
    dt = np.mean(np.diff(time))
    if dt == 0: return np.zeros_like(angle)
    return np.gradient(np.deg2rad(angle), dt) * STICK_LENGTH

def plot_exp4(df_real, df_sim_a, df_sim_b, out_dir):
    """Drumming Task"""
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    
    # Angle
    ax = axes[0]
    if df_real is not None:
        ax.plot(df_real['time'], df_real['angle_deg'], 'k-', label=REAL_LABEL)
    if df_sim_a is not None:
        ax.plot(df_sim_a['time'], df_sim_a['angle_deg'], 'b--', label=MODEL_A_LABEL)
    if df_sim_b is not None:
        ax.plot(df_sim_b['time'], df_sim_b['angle_deg'], 'r-.', label=MODEL_B_LABEL)
    ax.set_ylabel('Angle [deg]')
    ax.set_title('Exp 4: Drumming Task Performance')
    ax.legend()
    ax.grid(True)
    
    # Impact Force / Velocity
    ax = axes[1]
    if df_real is not None:
        if 'force_N' in df_real.columns and df_real['force_N'].max() > 0.1:
            ax.plot(df_real['time'], df_real['force_N'], 'k-', label='Measured Force [N]')

    if df_sim_a is not None:
        v = calculate_velocity_proxy(df_sim_a['time'], df_sim_a['force_z'])
        ax.plot(df_sim_a['time'], v, 'b--', alpha=0.6, label='Force (Model A)')

    if df_sim_b is not None:
        v = calculate_velocity_proxy(df_sim_b['time'], df_sim_b['force_z'])
        ax.plot(df_sim_b['time'], v, 'r-.', alpha=0.6, label='Force (Model B)')
        
    ax.set_ylabel('Impact Intensity')
    ax.legend()
    ax.grid(True)
    ax.set_xlabel('Time [s]')
    
    plt.savefig(os.path.join(out_dir, "exp4_drumming_comparison.png"))
    print("Saved: exp4_drumming_comparison.png")

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--real", type=str, required=True, help="Real data CSV")
    parser.add_argument("--sim_a", type=str, default=None, help="Model A Simulation CSV")
    parser.add_argument("--sim_b", type=str, default=None, help="Model B Simulation CSV")
    args = parser.parse_args()

    real_path = get_real_data_path(args.real)
    
    df_real = load_data(real_path, "REAL")
    df_sim_a = load_data(args.sim_a, "SIM_A")
    df_sim_b = load_data(args.sim_b, "SIM_B")
    
    out_dir = "analysis_results_iros_comparison"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"--- 3-Way Comparison: Exp {args.exp} ---")

    if args.exp == 1:
        plot_exp1(df_real, df_sim_a, df_sim_b, out_dir)
    elif args.exp == 2:
        plot_exp2(df_real, df_sim_a, df_sim_b, out_dir)
    elif args.exp == 3:
        plot_exp3(df_real, df_sim_a, df_sim_b, out_dir)
    elif args.exp == 4:
        plot_exp4(df_real, df_sim_a, df_sim_b, out_dir)