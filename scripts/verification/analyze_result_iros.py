"""
Porcaro Robot: IROS 2026 Final Analysis Script (v12)
Target: Exp1-8 Full Calibration Suite
Author: Robo-Dev Partner
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Settings
# ==========================================
def set_pub_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['savefig.dpi'] = 300

set_pub_style()
OUTPUT_DIR = "analysis_results_iros_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors
CLR_CMD = 'black'; CLR_REAL_MEAS = '#D62728'; CLR_SIM_OUT = '#1F77B4'; CLR_SIM_INT = '#2CA02C'

# ==========================================
# 2. File Finding & Loading
# ==========================================
def find_latest_pair(exp_keywords):
    search_root = "."
    real_cands, sim_cands = [], []
    for root, _, files in os.walk(search_root):
        for f in files:
            if not f.endswith(".csv") or "analysis" in root: continue
            path = os.path.join(root, f)
            if all(k in f for k in exp_keywords):
                if "sim_log" in f: sim_cands.append(path)
                elif "data_" in f: real_cands.append(path)

    if not real_cands:
        print(f"[Skip] No Real data for {exp_keywords}")
        return None, None
    
    # data_ を優先し最新を取得
    real_cands.sort(key=lambda x: (0 if "data_" in os.path.basename(x) else 1, -os.path.getctime(x)))
    real = real_cands[0]
    sim = max(sim_cands, key=os.path.getctime) if sim_cands else None
    print(f"[{exp_keywords[0]}] Pair:\n  Real: {os.path.basename(real)}\n  Sim : {os.path.basename(sim) if sim else 'None'}")
    return real, sim

def load_and_sync(real_path, sim_path):
    # Real
    try: df_r = pd.read_csv(real_path)
    except: return None, None
    if 'angle_deg' not in df_r.columns: return None, None
    
    t_col = next((c for c in ['time','timestamp','Time'] if c in df_r.columns), df_r.columns[0])
    df_r['t_sync'] = df_r[t_col] - df_r[t_col].iloc[0]
    
    # Missing Cols
    if 'meas_pres_DF' not in df_r.columns: df_r['meas_pres_DF'] = np.nan
    if 'force_N' not in df_r.columns: df_r['force_N'] = 0.0
    
    if not sim_path: return df_r, None

    # Sim
    try: df_s = pd.read_csv(sim_path)
    except: return df_r, None
    
    t_col_s = next((c for c in ['time','timestamp','Time'] if c in df_s.columns), df_s.columns[0])
    df_s['t_sync'] = df_s[t_col_s] - df_s[t_col_s].iloc[0]
    
    if 'sim_force_n' not in df_s.columns: df_s['sim_force_n'] = 0.0
    if 'sim_pres_DF' not in df_s.columns: df_s['sim_pres_DF'] = df_s.get('cmd_DF', np.nan)
    
    return df_r, df_s

# ==========================================
# 3. Plotting Functions
# ==========================================
def plot_generic_drumming(real_path, sim_path, title_prefix, filename_suffix):
    """ Exp4-8用の汎用ドラミングプロット (Angle, Pressure, Force) """
    df_r, df_s = load_and_sync(real_path, sim_path)
    if df_r is None: return

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # (a) Angle
    axes[0].plot(df_r['t_sync'], df_r['angle_deg'], color=CLR_REAL_MEAS, label='Real')
    if df_s is not None:
        axes[0].plot(df_s['t_sync'], df_s['sim_angle_deg'], color=CLR_SIM_OUT, linestyle='--', label='Sim')
    axes[0].set_ylabel('Angle [deg]'); axes[0].set_title(f'{title_prefix}: Angle'); axes[0].legend(loc='upper right'); axes[0].grid(True)

    # (b) Pressure
    axes[1].plot(df_r['t_sync'], df_r['meas_pres_DF'], color=CLR_REAL_MEAS, alpha=0.7, label='Real Meas')
    if df_s is not None:
        axes[1].plot(df_s['t_sync'], df_s['sim_pres_DF'], color=CLR_SIM_INT, linestyle='--', label='Sim Internal')
    axes[1].set_ylabel('Pressure [MPa]'); axes[1].set_title(f'{title_prefix}: Pressure'); axes[1].legend(loc='upper right'); axes[1].grid(True)

    # (c) Force
    axes[2].plot(df_r['t_sync'], df_r['force_N'], color=CLR_REAL_MEAS, label='Real Force')
    if df_s is not None:
        axes[2].plot(df_s['t_sync'], df_s['sim_force_n'], color=CLR_SIM_OUT, linestyle='--', label='Sim Force')
    axes[2].set_ylabel('Force [N]'); axes[2].set_xlabel('Time [s]'); axes[2].set_title(f'{title_prefix}: Force'); axes[2].legend(loc='upper right'); axes[2].grid(True)

    out = os.path.join(OUTPUT_DIR, f"Fig_{filename_suffix}.png")
    plt.savefig(out); plt.close()
    print(f"Saved: {out}")

# Wrappers
def plot_exp4(r, s): plot_generic_drumming(r, s, "Exp 4 (Drumming)", "Exp4_Drumming")
def plot_exp5(r, s): plot_generic_drumming(r, s, "Exp 5 (Amplitude)", "Exp5_Amplitude")
def plot_exp6(r, s): plot_generic_drumming(r, s, "Exp 6 (Duration)", "Exp6_Duration")
def plot_exp7(r, s): plot_generic_drumming(r, s, "Exp 7 (Stiffness)", "Exp7_Stiffness")
def plot_exp8(r, s): plot_generic_drumming(r, s, "Exp 8 (Speed)", "Exp8_Speed")

# Exp1-3 Functions (Simplified for brevity, same as before)
def plot_exp1(r, s): # Hysteresis
    df_r, df_s = load_and_sync(r, s)
    if df_r is None: return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(df_r['t_sync'], df_r['meas_pres_DF'], color=CLR_REAL_MEAS, label='Real')
    if df_s is not None: axes[0].plot(df_s['t_sync'], df_s['sim_pres_DF'], color=CLR_SIM_INT, ls='--', label='Sim')
    axes[0].set_title('Pressure'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(df_r['meas_pres_DF'], df_r['angle_deg'], color=CLR_REAL_MEAS)
    if df_s is not None: axes[1].plot(df_s['sim_pres_DF'], df_s['sim_angle_deg'], color=CLR_SIM_OUT, ls='--')
    axes[1].set_title('Hysteresis Loop'); axes[1].grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_Exp1_Hysteresis.png")); plt.close()

def plot_exp2(r, s): # Step
    plot_generic_drumming(r, s, "Exp 2 (Step)", "Exp2_Step") # Reuse generic for simplicity

def plot_exp3(r, s): # Sweep
    df_r, df_s = load_and_sync(r, s)
    if df_r is None: return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_r['t_sync'], df_r['angle_deg'], color=CLR_REAL_MEAS, label='Real')
    if df_s is not None: ax.plot(df_s['t_sync'], df_s['sim_angle_deg'], color=CLR_SIM_OUT, ls='--', label='Sim')
    ax.set_title('Exp 3: Frequency Sweep'); ax.legend(); ax.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_Exp3_Sweep.png")); plt.close()

# ==========================================
# 4. Main
# ==========================================
if __name__ == "__main__":
    exps = [
        (["exp1", "static"], plot_exp1, "Exp 1"),
        (["exp2", "step"],   plot_exp2, "Exp 2"),
        (["exp3", "sweep"],  plot_exp3, "Exp 3"),
        (["exp4", "drum"],   plot_exp4, "Exp 4"),
        (["exp5", "amp"],    plot_exp5, "Exp 5"),
        (["exp6", "dur"],    plot_exp6, "Exp 6"),
        (["exp7", "stiff"],  plot_exp7, "Exp 7"),
        (["exp8", "speed"],  plot_exp8, "Exp 8"),
    ]

    for keywords, func, label in exps:
        print(f"\n=== Analyzing {label} ===")
        r, s = find_latest_pair(keywords)
        if r: func(r, s)
    
    print(f"\nAnalysis Completed. Check '{OUTPUT_DIR}'")