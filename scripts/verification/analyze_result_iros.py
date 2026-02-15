"""
Porcaro Robot: IROS 2026 Final Analysis Script (v8)
Target: Simple Raw Comparison (Zero-Start Alignment)
Author: Robo-Dev Partner
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Settings & Constants
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
CLR_CMD       = 'black'
CLR_REAL_MEAS = '#D62728'  # Red
CLR_SIM_OUT   = '#1F77B4'  # Blue
CLR_SIM_INT   = '#2CA02C'  # Green

# ==========================================
# 2. Robust File Finding Logic
# ==========================================
def find_latest_pair(exp_keywords):
    """ 指定キーワードを含む最新のログペアを探す """
    search_root = "."
    real_candidates = []
    sim_candidates = []

    for root, dirs, files in os.walk(search_root):
        for file in files:
            if not file.endswith(".csv"): continue
            if "analysis_results" in root: continue 
            
            path = os.path.join(root, file)
            if all(k in file for k in exp_keywords):
                if "sim_log" in file:
                    sim_candidates.append(path)
                elif "data_" in file or "exp" in file:
                    real_candidates.append(path)

    if not real_candidates:
        print(f"[Skip] No Real data found for keywords: {exp_keywords}")
        return None, None
    
    real_file = max(real_candidates, key=os.path.getctime)
    sim_file = max(sim_candidates, key=os.path.getctime) if sim_candidates else None
    
    print(f"[{exp_keywords[0]}] Pair Found:\n  Real: {os.path.basename(real_file)}\n  Sim : {os.path.basename(sim_file) if sim_file else 'None'}")
    return real_file, sim_file

# ==========================================
# 3. Simple Data Loading (Zero-Start Alignment)
# ==========================================
def load_and_sync(real_path, sim_path):
    """ 
    同期計算を行わず、単に0秒から開始するように正規化するだけの関数。
    """
    # --- Load Real ---
    df_real = pd.read_csv(real_path)
    t_col_r = next((c for c in ['time', 'timestamp', 'Time'] if c in df_real.columns), df_real.columns[0])
    # 変更箇所: 最初のタイムスタンプを引いて0秒スタートにする
    df_real['t_sync'] = df_real[t_col_r] - df_real[t_col_r].iloc[0]
    
    if 'force_N' not in df_real.columns: df_real['force_N'] = 0.0
    if 'meas_pres_DF' not in df_real.columns: df_real['meas_pres_DF'] = np.nan

    if not sim_path:
        return df_real, None

    # --- Load Sim ---
    df_sim = pd.read_csv(sim_path)
    t_col_s = next((c for c in ['time', 'timestamp', 'Time'] if c in df_sim.columns), df_sim.columns[0])
    # 変更箇所: 最初のタイムスタンプを引いて0秒スタートにする
    df_sim['t_sync'] = df_sim[t_col_s] - df_sim[t_col_s].iloc[0]

    # カラムの補完
    if 'sim_force_n' not in df_sim.columns: df_sim['sim_force_n'] = 0.0
    if 'sim_pres_DF' not in df_sim.columns:
        df_sim['sim_pres_DF'] = df_sim['cmd_DF'] if 'cmd_DF' in df_sim.columns else np.nan
    
    print(f"  -> Comparison started from t=0.0s (Raw Alignment)")
    return df_real, df_sim

# ==========================================
# 4. Plotting Functions
# ==========================================

def plot_exp1_hysteresis(real_path, sim_path):
    df_r, df_s = load_and_sync(real_path, sim_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(df_r['t_sync'], df_r['meas_pres_DF'], color=CLR_REAL_MEAS, label='Real Meas')
    if df_s is not None:
        axes[0].plot(df_s['t_sync'], df_s['sim_pres_DF'], color=CLR_SIM_INT, linestyle='--', label='Sim Internal')
    axes[0].set_xlabel('Time [s]'); axes[0].set_ylabel('Pressure [MPa]'); axes[0].legend(); axes[0].grid(True)
    
    axes[1].plot(df_r['meas_pres_DF'], df_r['angle_deg'], color=CLR_REAL_MEAS, label='Real')
    if df_s is not None:
        axes[1].plot(df_s['sim_pres_DF'], df_s['sim_angle_deg'], color=CLR_SIM_OUT, linestyle='--', label='Sim')
    axes[1].set_xlabel('Pressure [MPa]'); axes[1].set_ylabel('Angle [deg]'); axes[1].legend(); axes[1].grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_Exp1_Hysteresis.png")); plt.close()

def plot_exp2_step(real_path, sim_path):
    df_r, df_s = load_and_sync(real_path, sim_path)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(df_r['t_sync'], df_r['angle_deg'], color=CLR_REAL_MEAS, label='Real')
    if df_s is not None: axes[0].plot(df_s['t_sync'], df_s['sim_angle_deg'], color=CLR_SIM_OUT, linestyle='--', label='Sim')
    axes[0].set_ylabel('Angle [deg]'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(df_r['t_sync'], df_r['meas_pres_DF'], color=CLR_REAL_MEAS, label='Real Meas')
    if df_s is not None: axes[1].plot(df_s['t_sync'], df_s['sim_pres_DF'], color=CLR_SIM_INT, linestyle='--', label='Sim Internal')
    axes[1].set_ylabel('Pressure [MPa]'); axes[1].set_xlabel('Time [s]'); axes[1].legend(); axes[1].grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_Exp2_Step.png")); plt.close()

def plot_exp3_analysis(real_path, sim_path):
    df_r, df_s = load_and_sync(real_path, sim_path)
    if df_s is None: return
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(df_r['t_sync'], df_r['angle_deg'], color=CLR_REAL_MEAS, label='Real'); axes[0].plot(df_s['t_sync'], df_s['sim_angle_deg'], color=CLR_SIM_OUT, linestyle='--', label='Sim')
    axes[1].plot(df_r['t_sync'], df_r['meas_pres_DF'], color=CLR_REAL_MEAS, label='Real Meas'); axes[1].plot(df_s['t_sync'], df_s['sim_pres_DF'], color=CLR_SIM_INT, linestyle='--', label='Sim Internal')
    axes[0].set_ylabel('Angle [deg]'); axes[1].set_ylabel('Pressure [MPa]'); axes[1].set_xlabel('Time [s]'); axes[0].grid(True); axes[1].grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_Exp3_TimeSeries.png")); plt.close()

def plot_exp4_drumming(real_path, sim_path):
    """ Exp-4: Drumming Task (Angle, Pressure, Force の3軸比較) """
    df_r, df_s = load_and_sync(real_path, sim_path)
    if df_s is None: return

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # (a) Angle
    axes[0].plot(df_r['t_sync'], df_r['angle_deg'], color=CLR_REAL_MEAS, label='Real')
    axes[0].plot(df_s['t_sync'], df_s['sim_angle_deg'], color=CLR_SIM_OUT, linestyle='--', label='Sim')
    axes[0].set_ylabel('Angle [deg]'); axes[0].set_title('Drumming: Angle Response'); axes[0].legend(loc='upper right'); axes[0].grid(True)

    # (b) Pressure
    axes[1].plot(df_r['t_sync'], df_r['meas_pres_DF'], color=CLR_REAL_MEAS, alpha=0.7, label='Real Meas')
    axes[1].plot(df_s['t_sync'], df_s['sim_pres_DF'], color=CLR_SIM_INT, linestyle='--', label='Sim Internal')
    axes[1].set_ylabel('Pressure [MPa]'); axes[1].set_title('Drumming: Pressure Dynamics'); axes[1].legend(loc='upper right'); axes[1].grid(True)

    # (c) Contact Force
    axes[2].plot(df_r['t_sync'], df_r['force_N'], color=CLR_REAL_MEAS, label='Real Force')
    axes[2].plot(df_s['t_sync'], df_s['sim_force_n'], color=CLR_SIM_OUT, linestyle='--', label='Sim Force')
    axes[2].set_ylabel('Force [N]'); axes[2].set_xlabel('Time [s]'); axes[2].set_title('Drumming: Contact Force'); axes[2].legend(loc='upper right'); axes[2].grid(True)

    out = os.path.join(OUTPUT_DIR, "Fig_Exp4_Drumming.png")
    plt.savefig(out); plt.close()
    print(f"Saved: {out}")

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    exps = [
        (["exp1", "static", "hysteresis"], plot_exp1_hysteresis, "Exp 1"),
        (["exp2", "step"], plot_exp2_step, "Exp 2"),
        (["exp3", "sweep", "freq"], plot_exp3_analysis, "Exp 3"),
        (["exp4", "drumming", "task"], plot_exp4_drumming, "Exp 4")
    ]

    for keywords, func, label in exps:
        print(f"\n=== Analyzing {label} ===")
        r, s = find_latest_pair(keywords)
        if r: func(r, s)
    
    print(f"\nAnalysis Completed. Check '{OUTPUT_DIR}'")