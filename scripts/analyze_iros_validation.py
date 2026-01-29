# scripts/analyze_iros_validation.py
"""
Porcaro Robot: IROS 2026 Validation Analysis Script (Full Suite)
Comparison: Physical System vs Model A vs Model B
Features: 
  - Exp1: Static Hysteresis Loop
  - Exp2: Step Response
  - Exp3: Frequency Sweep (Angle & Pressure with RMSE) -> UPDATED
  - Exp4: Drumming Task
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
REAL_LABEL = "Physical System"
MODEL_A_LABEL = "Model A (Baseline)"
MODEL_B_LABEL = "Model B (Proposed)"

STICK_LENGTH = 0.35
MAX_PRESSURE = 0.6

# プロットスタイル
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['figure.dpi'] = 300

def load_data(path, label, is_sim=False):
    """CSV読み込み・整形"""
    if not path or not os.path.exists(path):
        if path: print(f"[WARN] File not found: {path} (Label: {label})")
        return None
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        return None
    
    # カラム名統一
    rename_map = {
        'time_s': 'time', 'timestamp': 'time', 'Time': 'time',
        'q_wrist_deg': 'angle', 'angle_deg': 'angle', 
        'obs_angle': 'angle', 'obs_joint_pos': 'angle', 'joint_pos': 'angle',
        'P_cmd_DF': 'pressure_cmd', 'cmd_DF': 'pressure_cmd', 
        'cmd_pressure_DF': 'pressure_cmd',
        'P_out_DF': 'pressure_meas', 'meas_pres_DF': 'pressure_meas',
        'force_z': 'force', 'force_N': 'force'
    }
    df = df.rename(columns=rename_map)
    
    # 時間軸正規化
    if 'time' in df.columns:
        df['time'] = df['time'] - df['time'].iloc[0]
    else:
        df['time'] = np.arange(len(df)) * 0.02

    # 圧力データの復元・補完
    if 'pressure_cmd' not in df.columns and 'action_0' in df.columns:
        df['pressure_cmd'] = (df['action_0'] + 1.0) / 2.0 * MAX_PRESSURE

    if 'pressure_meas' not in df.columns:
        if 'pressure_cmd' in df.columns:
            df['pressure_meas'] = df['pressure_cmd']

    return df

def get_real_data_path(filename):
    """データ探索ロジック"""
    if os.path.exists(filename): return filename
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    candidates = [
        os.path.join(project_root, "external_data", "jetson_project", "IROS", "test_signals", filename),
        os.path.join(project_root, "IROS", "test_signals", filename),
        os.path.join("data", filename),
        os.path.join("test_signals", filename),
        filename
    ]
    for path in candidates:
        if os.path.exists(path): return path
    return filename

def calc_rmse(y_true, y_pred):
    n = min(len(y_true), len(y_pred))
    if n == 0: return 0.0
    return np.sqrt(mean_squared_error(y_true[:n], y_pred[:n]))

def calculate_velocity_proxy(time, angle):
    dt = np.mean(np.diff(time))
    if dt == 0: return np.zeros_like(angle)
    return np.gradient(np.deg2rad(angle), dt) * STICK_LENGTH

# ==========================================
# Plotting Functions
# ==========================================

def plot_exp1(df_real, df_sim_a, df_sim_b, out_dir):
    """Exp 1: Static Hysteresis Loop"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Hysteresis Loop
    ax = axes[0]
    if df_real is not None:
        ax.plot(df_real['pressure_meas'], df_real['angle'], 'k-', label=REAL_LABEL, linewidth=2.5, alpha=0.8)
    if df_sim_a is not None:
        ax.plot(df_sim_a['pressure_meas'], df_sim_a['angle'], 'b:', label=MODEL_A_LABEL)
    if df_sim_b is not None:
        ax.plot(df_sim_b['pressure_meas'], df_sim_b['angle'], 'r-.', label=MODEL_B_LABEL)
    
    ax.set_xlabel('Pressure [MPa]')
    ax.set_ylabel('Angle [deg]')
    ax.set_title('Exp 1: Hysteresis Loop')
    ax.legend()
    ax.grid(True)

    # Right: Error Distribution
    ax = axes[1]
    if df_real is not None:
        real_p = df_real['pressure_meas'].values
        real_a = df_real['angle'].values
        if df_sim_a is not None:
            n = min(len(real_a), len(df_sim_a))
            err = real_a[:n] - df_sim_a['angle'].values[:n]
            ax.plot(real_p[:n], np.abs(err), 'b.', markersize=2, alpha=0.3, label='Error A')
        if df_sim_b is not None:
            n = min(len(real_a), len(df_sim_b))
            err = real_a[:n] - df_sim_b['angle'].values[:n]
            ax.plot(real_p[:n], np.abs(err), 'r.', markersize=2, alpha=0.3, label='Error B')

    ax.set_xlabel('Pressure [MPa]')
    ax.set_ylabel('Abs Angle Error [deg]')
    ax.set_title('Error vs Pressure')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp1_hysteresis_comparison.png"))
    print("Saved: exp1_hysteresis_comparison.png")

def plot_exp2(df_real, df_sim_a, df_sim_b, out_dir):
    """Exp 2: Step Response"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1.5, 1]})
    
    # Angle
    ax = axes[0]
    if df_real is not None: ax.plot(df_real['time'], df_real['angle'], 'k-', label=REAL_LABEL)
    if df_sim_a is not None: ax.plot(df_sim_a['time'], df_sim_a['angle'], 'b--', label=MODEL_A_LABEL)
    if df_sim_b is not None: ax.plot(df_sim_b['time'], df_sim_b['angle'], 'r-.', label=MODEL_B_LABEL)
    ax.set_ylabel('Angle [deg]')
    ax.legend()
    ax.grid(True)
    
    # Pressure
    ax = axes[1]
    if df_real is not None: ax.plot(df_real['time'], df_real['pressure_meas'], 'k:', label='Real Meas')
    if df_sim_a is not None: ax.plot(df_sim_a['time'], df_sim_a['pressure_meas'], 'b--', label='Sim A Internal')
    if df_sim_b is not None: ax.plot(df_sim_b['time'], df_sim_b['pressure_meas'], 'r-.', label='Sim B Internal')
    ax.set_ylabel('Pressure [MPa]')
    ax.legend()
    ax.grid(True)

    # Error
    ax = axes[2]
    t = df_real['time'].values if df_real is not None else []
    if df_real is not None and df_sim_a is not None:
        n = min(len(t), len(df_sim_a))
        ax.plot(t[:n], np.abs(df_real['angle'].values[:n] - df_sim_a['angle'].values[:n]), 'b--', alpha=0.5, label='Error A')
    if df_real is not None and df_sim_b is not None:
        n = min(len(t), len(df_sim_b))
        ax.plot(t[:n], np.abs(df_real['angle'].values[:n] - df_sim_b['angle'].values[:n]), 'r-', alpha=0.6, label='Error B')
    ax.set_ylabel('Abs Error [deg]')
    ax.set_xlabel('Time [s]')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp2_step_comparison.png"))
    print("Saved: exp2_step_comparison.png")

def plot_exp3(df_real, df_sim_a, df_sim_b, out_dir):
    """
    Exp 3: Frequency Sweep - UPDATED
    Rows: Angle, Angle Error, Pressure, Pressure Error
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True, 
                             gridspec_kw={'height_ratios': [2, 1, 2, 1]})
    
    t_real = df_real['time'].values if df_real is not None else []
    
    # --- 1. Angle ---
    ax = axes[0]
    real_ang = None
    if df_real is not None:
        ax.plot(df_real['time'], df_real['angle'], 'k-', label=REAL_LABEL, alpha=0.8)
        real_ang = df_real['angle'].values

    rmse_ang_a = 0.0
    if df_sim_a is not None and real_ang is not None:
        rmse_ang_a = calc_rmse(real_ang, df_sim_a['angle'].values)
        ax.plot(df_sim_a['time'], df_sim_a['angle'], 'b--', label=f"{MODEL_A_LABEL} (RMSE={rmse_ang_a:.2f})")
    
    rmse_ang_b = 0.0
    if df_sim_b is not None and real_ang is not None:
        rmse_ang_b = calc_rmse(real_ang, df_sim_b['angle'].values)
        ax.plot(df_sim_b['time'], df_sim_b['angle'], 'r-.', label=f"{MODEL_B_LABEL} (RMSE={rmse_ang_b:.2f})")
    
    ax.set_ylabel('Angle [deg]')
    ax.set_title('Exp 3: Frequency Sweep - Angle Tracking')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True)

    # --- 2. Angle Error ---
    ax = axes[1]
    if df_real is not None:
        if df_sim_a is not None:
            n = min(len(t_real), len(df_sim_a))
            ax.plot(t_real[:n], np.abs(df_real['angle'].values[:n] - df_sim_a['angle'].values[:n]), 'b--', alpha=0.5)
        if df_sim_b is not None:
            n = min(len(t_real), len(df_sim_b))
            ax.plot(t_real[:n], np.abs(df_real['angle'].values[:n] - df_sim_b['angle'].values[:n]), 'r-', alpha=0.6)
    ax.set_ylabel('Angle Error')
    ax.grid(True)

    # --- 3. Pressure ---
    ax = axes[2]
    real_pres = None
    if df_real is not None:
        ax.plot(df_real['time'], df_real['pressure_meas'], 'k-', label='Real Meas', alpha=0.8)
        real_pres = df_real['pressure_meas'].values

    rmse_pres_a = 0.0
    if df_sim_a is not None and real_pres is not None:
        rmse_pres_a = calc_rmse(real_pres, df_sim_a['pressure_meas'].values)
        ax.plot(df_sim_a['time'], df_sim_a['pressure_meas'], 'b--', label=f"Sim A (RMSE={rmse_pres_a:.3f})")

    rmse_pres_b = 0.0
    if df_sim_b is not None and real_pres is not None:
        rmse_pres_b = calc_rmse(real_pres, df_sim_b['pressure_meas'].values)
        ax.plot(df_sim_b['time'], df_sim_b['pressure_meas'], 'r-.', label=f"Sim B (RMSE={rmse_pres_b:.3f})")

    ax.set_ylabel('Pressure [MPa]')
    ax.set_title('Pressure Dynamics Tracking')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True)

    # --- 4. Pressure Error ---
    ax = axes[3]
    if df_real is not None:
        if df_sim_a is not None:
            n = min(len(t_real), len(df_sim_a))
            ax.plot(t_real[:n], np.abs(df_real['pressure_meas'].values[:n] - df_sim_a['pressure_meas'].values[:n]), 'b--', alpha=0.5)
        if df_sim_b is not None:
            n = min(len(t_real), len(df_sim_b))
            ax.plot(t_real[:n], np.abs(df_real['pressure_meas'].values[:n] - df_sim_b['pressure_meas'].values[:n]), 'r-', alpha=0.6)
    ax.set_ylabel('Pres Error')
    ax.set_xlabel('Time [s]')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp3_sweep_comparison.png"))
    print("Saved: exp3_sweep_comparison.png")
    
    # Console Output
    print(f"\n[Exp 3 RMSE Report]")
    print(f"Angle:    Model A={rmse_ang_a:.2f}, Model B={rmse_ang_b:.2f}")
    print(f"Pressure: Model A={rmse_pres_a:.3f}, Model B={rmse_pres_b:.3f}\n")

def plot_exp4(df_real, df_sim_a, df_sim_b, out_dir):
    """Exp 4: Drumming Task"""
    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True, gridspec_kw={'height_ratios': [2, 1.5, 1.5, 1]})
    
    # Angle
    ax = axes[0]
    if df_real is not None: ax.plot(df_real['time'], df_real['angle'], 'k-', label=REAL_LABEL)
    if df_sim_a is not None: ax.plot(df_sim_a['time'], df_sim_a['angle'], 'b--', label=MODEL_A_LABEL)
    if df_sim_b is not None: ax.plot(df_sim_b['time'], df_sim_b['angle'], 'r-.', label=MODEL_B_LABEL)
    ax.set_ylabel('Angle [deg]')
    ax.legend()
    ax.grid(True)

    # Pressure
    ax = axes[1]
    if df_real is not None: ax.plot(df_real['time'], df_real['pressure_meas'], 'k-', label='Real Meas')
    if df_sim_b is not None: ax.plot(df_sim_b['time'], df_sim_b['pressure_meas'], 'r-.', label='Sim B Internal')
    ax.set_ylabel('Pressure [MPa]')
    ax.legend()
    ax.grid(True)
    
    # Force
    ax = axes[2]
    if df_real is not None and 'force' in df_real.columns:
        ax.plot(df_real['time'], df_real['force'], 'k-', label='Real Sensor')
    if df_sim_a is not None:
        # Use force column if exists, else velocity proxy
        if 'force' in df_sim_a.columns:
             ax.plot(df_sim_a['time'], df_sim_a['force'], 'b--', label='Sim A Force')
        else:
             v = calculate_velocity_proxy(df_sim_a['time'], df_sim_a['angle'])
             ax.plot(df_sim_a['time'], v, 'b--', alpha=0.5, label='Sim A Vel Proxy')
    if df_sim_b is not None:
        if 'force' in df_sim_b.columns:
             ax.plot(df_sim_b['time'], df_sim_b['force'], 'r-.', label='Sim B Force')
        else:
             v = calculate_velocity_proxy(df_sim_b['time'], df_sim_b['angle'])
             ax.plot(df_sim_b['time'], v, 'r-.', alpha=0.5, label='Sim B Vel Proxy')
    ax.set_ylabel('Impact [N]')
    ax.legend()
    ax.grid(True)

    # Error
    ax = axes[3]
    t = df_real['time'].values if df_real is not None else []
    if df_real is not None and df_sim_a is not None:
        n = min(len(t), len(df_sim_a))
        ax.plot(t[:n], np.abs(df_real['angle'].values[:n] - df_sim_a['angle'].values[:n]), 'b--', alpha=0.5, label='Error A')
    if df_real is not None and df_sim_b is not None:
        n = min(len(t), len(df_sim_b))
        ax.plot(t[:n], np.abs(df_real['angle'].values[:n] - df_sim_b['angle'].values[:n]), 'r-', alpha=0.6, label='Error B')
    ax.set_ylabel('Abs Error')
    ax.set_xlabel('Time [s]')
    ax.grid(True)
    plt.tight_layout()
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