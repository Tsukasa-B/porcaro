"""
Porcaro Robot: IROS 2026 Final Analysis Script (v4)
Target: Full Comparison with Command Pressure Reference
Author: Robo-Dev Partner
"""
import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

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
CLR_CMD       = 'black'    # Command (Target)
CLR_REAL_MEAS = '#D62728'  # Red (Real Angle / Meas Pressure)
CLR_SIM_OUT   = '#1F77B4'  # Blue (Sim Angle)
CLR_SIM_INT   = '#2CA02C'  # Green (Sim Internal Pressure)

# ==========================================
# 2. Robust File Finding Logic
# ==========================================
def find_latest_pair(exp_keywords):
    """ 指定キーワードを含む最新のログペアを探す (再帰検索対応) """
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
    
    print(f"[{exp_keywords[0]}] Pair:\n  Real: {os.path.basename(real_file)}\n  Sim : {os.path.basename(sim_file) if sim_file else 'None'}")
    return real_file, sim_file

# ==========================================
# 3. Data Loading & Sync
# ==========================================
def load_and_sync(real_path, sim_path):
    """ 実機とSimのロード・同期 """
    # --- Load Real ---
    df_real = pd.read_csv(real_path)
    t_col = next((c for c in ['time', 'timestamp', 'Time'] if c in df_real.columns), df_real.columns[0])
    df_real['t'] = df_real[t_col] - df_real[t_col].iloc[0]
    
    # Column mapping
    if 'meas_pres_DF' not in df_real.columns: df_real['meas_pres_DF'] = np.nan
    if 'cmd_pressure_DF' not in df_real.columns:
         df_real['cmd_pressure_DF'] = df_real['cmd_DF'] if 'cmd_DF' in df_real.columns else np.nan

    if not sim_path:
        df_real['t_sync'] = df_real['t']
        return df_real, None

    # --- Load Sim ---
    df_sim = pd.read_csv(sim_path)
    t_col_s = next((c for c in ['time', 'timestamp', 'Time'] if c in df_sim.columns), df_sim.columns[0])
    df_sim['t'] = df_sim[t_col_s] - df_sim[t_col_s].iloc[0]

    # Check for Internal Pressure Log
    if 'sim_pres_DF' not in df_sim.columns:
        if 'cmd_DF' in df_sim.columns:
             df_sim['sim_pres_DF'] = df_sim['cmd_DF']
        else:
             df_sim['sim_pres_DF'] = np.nan
             
    # Ensure cmd_DF exists for plotting
    if 'cmd_DF' not in df_sim.columns:
        df_sim['cmd_DF'] = np.nan

    # --- Sync (Cross-Correlation on Angle) ---
    dt = 0.005
    t_max = min(df_real['t'].max(), df_sim['t'].max())
    t_common = np.arange(0, t_max, dt)
    
    ang_r = np.interp(t_common, df_real['t'], df_real['angle_deg'])
    ang_s = np.interp(t_common, df_sim['t'], df_sim['sim_angle_deg'])
    
    correlation = correlate(ang_r - np.mean(ang_r), ang_s - np.mean(ang_s), mode='full')
    lags = correlation_lags(ang_r.size, ang_s.size, mode='full')
    lag_idx = lags[np.argmax(correlation)]
    time_shift = lag_idx * dt
    
    print(f"  -> Sync Shift: {time_shift:.4f} s")
    
    df_sim['t_sync'] = df_sim['t'] + time_shift
    df_real['t_sync'] = df_real['t']
    
    return df_real, df_sim

def calc_loop_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# ==========================================
# 4. Plotting Functions (Updated with Command)
# ==========================================

def plot_exp1_hysteresis(real_path, sim_path):
    """ Exp-1: 圧力 vs 角度 (Command追加) """
    df_r, df_s = load_and_sync(real_path, sim_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    
    # (a) Pressure Tracking
    ax1 = axes[0]
    # Command (Common Input)
    if df_s is not None and 'cmd_DF' in df_s.columns:
        ax1.plot(df_s['t_sync'], df_s['cmd_DF'], color=CLR_CMD, linestyle=':', linewidth=1.5, label='Command')
    elif 'cmd_pressure_DF' in df_r.columns:
        ax1.plot(df_r['t_sync'], df_r['cmd_pressure_DF'], color=CLR_CMD, linestyle=':', linewidth=1.5, label='Command')

    # Responses
    ax1.plot(df_r['t_sync'], df_r['meas_pres_DF'], color=CLR_REAL_MEAS, label='Real Meas')
    if df_s is not None:
        ax1.plot(df_s['t_sync'], df_s['sim_pres_DF'], color=CLR_SIM_INT, linestyle='--', label='Sim Internal')
    
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Pressure [MPa]')
    ax1.set_title('(a) Pressure Dynamics Check')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True)
    
    # (b) Hysteresis Loop
    ax2 = axes[1]
    ax2.plot(df_r['meas_pres_DF'], df_r['angle_deg'], color=CLR_REAL_MEAS, linewidth=2, label='Real')
    
    metrics_txt = ""
    if df_s is not None:
        ax2.plot(df_s['sim_pres_DF'], df_s['sim_angle_deg'], color=CLR_SIM_OUT, linestyle='--', linewidth=2, label='Sim')
        
        t_eval = np.linspace(df_r['t_sync'].min(), df_r['t_sync'].max(), 500)
        pr_i = np.interp(t_eval, df_r['t_sync'], df_r['meas_pres_DF'])
        ar_i = np.interp(t_eval, df_r['t_sync'], df_r['angle_deg'])
        ps_i = np.interp(t_eval, df_s['t_sync'], df_s['sim_pres_DF'])
        as_i = np.interp(t_eval, df_s['t_sync'], df_s['sim_angle_deg'])
        
        area_r = calc_loop_area(pr_i, ar_i)
        area_s = calc_loop_area(ps_i, as_i)
        ratio = area_s / area_r if area_r > 0 else 0
        metrics_txt = f"\nArea Ratio: {ratio:.2f}"

    ax2.set_xlabel('Pressure [MPa]')
    ax2.set_ylabel('Angle [deg]')
    ax2.set_title(f'(b) Hysteresis Loop {metrics_txt}')
    ax2.legend()
    ax2.grid(True)
    
    out = os.path.join(OUTPUT_DIR, "Fig_Exp1_Hysteresis.png")
    plt.savefig(out)
    print(f"Saved: {out}")
    plt.close()

def plot_exp2_step(real_path, sim_path):
    """ Exp-2: Step Response (Command追加) """
    df_r, df_s = load_and_sync(real_path, sim_path)
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    
    # (a) Angle
    ax1 = axes[0]
    ax1.plot(df_r['t_sync'], df_r['angle_deg'], color=CLR_REAL_MEAS, label='Real')
    if df_s is not None:
        ax1.plot(df_s['t_sync'], df_s['sim_angle_deg'], color=CLR_SIM_OUT, linestyle='--', label='Sim')
        
        t_eval = np.linspace(df_r['t_sync'].min(), df_r['t_sync'].max(), 1000)
        ar_i = np.interp(t_eval, df_r['t_sync'], df_r['angle_deg'])
        as_i = np.interp(t_eval, df_s['t_sync'], df_s['sim_angle_deg'])
        rmse = np.sqrt(np.mean((ar_i - as_i)**2))
        ax1.set_title(f'(a) Step Response (RMSE: {rmse:.2f} deg)')

    ax1.set_ylabel('Angle [deg]')
    ax1.legend()
    ax1.grid(True)
    
    # (b) Pressure Dynamics
    ax2 = axes[1]
    # Command
    if df_s is not None and 'cmd_DF' in df_s.columns:
        ax2.plot(df_s['t_sync'], df_s['cmd_DF'], color=CLR_CMD, linestyle=':', linewidth=1.5, label='Command')
    elif 'cmd_pressure_DF' in df_r.columns:
        ax2.plot(df_r['t_sync'], df_r['cmd_pressure_DF'], color=CLR_CMD, linestyle=':', linewidth=1.5, label='Command')

    # Real & Sim
    ax2.plot(df_r['t_sync'], df_r['meas_pres_DF'], color=CLR_REAL_MEAS, alpha=0.8, label='Real Meas')
    if df_s is not None:
        ax2.plot(df_s['t_sync'], df_s['sim_pres_DF'], color=CLR_SIM_INT, linestyle='--', label='Sim Internal')

    ax2.set_ylabel('Pressure [MPa]')
    ax2.set_xlabel('Time [s]')
    ax2.set_title('(b) Pressure Dynamics')
    ax2.legend(loc='lower right')
    ax2.grid(True)
    
    out = os.path.join(OUTPUT_DIR, "Fig_Exp2_Step.png")
    plt.savefig(out)
    print(f"Saved: {out}")
    plt.close()

def plot_exp3_analysis(real_path, sim_path):
    """ Exp-3: Frequency Sweep (Command追加) """
    df_r, df_s = load_and_sync(real_path, sim_path)
    if df_s is None: return

    # --- Figure 1: Time Domain ---
    fig1, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # (a) Angle
    ax1 = axes[0]
    ax1.plot(df_r['t_sync'], df_r['angle_deg'], color=CLR_REAL_MEAS, linewidth=1, label='Real Angle')
    ax1.plot(df_s['t_sync'], df_s['sim_angle_deg'], color=CLR_SIM_OUT, linewidth=1, linestyle='--', label='Sim Angle')
    ax1.set_ylabel('Angle [deg]')
    ax1.set_title('Exp-3 (Time Domain): Angle Response')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # (b) Pressure Tracking
    ax2 = axes[1]
    # Command
    if 'cmd_DF' in df_s.columns:
        ax2.plot(df_s['t_sync'], df_s['cmd_DF'], color=CLR_CMD, linestyle=':', linewidth=1, label='Command')
    
    # Real & Sim
    ax2.plot(df_r['t_sync'], df_r['meas_pres_DF'], color=CLR_REAL_MEAS, linewidth=1, alpha=0.8, label='Real Meas')
    ax2.plot(df_s['t_sync'], df_s['sim_pres_DF'], color=CLR_SIM_INT, linewidth=1, linestyle='--', label='Sim Internal')
    
    ax2.set_ylabel('Pressure [MPa]')
    ax2.set_xlabel('Time [s]')
    ax2.set_title('Exp-3 (Time Domain): Pressure Tracking')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    out1 = os.path.join(OUTPUT_DIR, "Fig_Exp3_TimeSeries.png")
    plt.savefig(out1)
    print(f"Saved: {out1}")
    plt.close(fig1)

    # --- Figure 2: Bode Plot (Unchanged) ---
    t_end = df_r['t_sync'].max()
    freq_start, freq_end = 0.1, 5.0
    def time_to_freq(t): return freq_start + (freq_end - freq_start) * (t / t_end)

    win = 50
    amp_r = df_r['angle_deg'].rolling(win, center=True).max() - df_r['angle_deg'].rolling(win, center=True).min()
    amp_s = df_s['sim_angle_deg'].rolling(win, center=True).max() - df_s['sim_angle_deg'].rolling(win, center=True).min()
    
    max_amp = amp_r.max()
    gain_r = 20 * np.log10(amp_r / max_amp + 1e-6)
    gain_s = 20 * np.log10(amp_s / max_amp + 1e-6)
    
    fig2, ax = plt.subplots(figsize=(7, 5))
    ax.plot(time_to_freq(df_r['t_sync']), gain_r, color=CLR_REAL_MEAS, label='Real Response')
    ax.plot(time_to_freq(df_s['t_sync']), gain_s, color=CLR_SIM_OUT, linestyle='--', label='Sim Response')
    ax.set_xscale('log')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Magnitude [dB]')
    ax.set_title('Exp-3 (Freq Domain): Bode Plot')
    ax.set_ylim(-20, 2)
    ax.set_xlim(0.1, 5.0)
    ax.axhline(-3, color='gray', linestyle=':', label='-3dB Bandwidth')
    ax.legend()
    ax.grid(True, which="both", ls="-")
    
    out2 = os.path.join(OUTPUT_DIR, "Fig_Exp3_Bode.png")
    plt.savefig(out2)
    print(f"Saved: {out2}")
    plt.close(fig2)

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    print("=== Analyzing Exp 1 (Hysteresis & Energy) ===")
    r, s = find_latest_pair(["exp1", "static", "hysteresis"])
    if r: plot_exp1_hysteresis(r, s)

    print("\n=== Analyzing Exp 2 (Step Response) ===")
    r, s = find_latest_pair(["exp2", "step"])
    if r: plot_exp2_step(r, s)

    print("\n=== Analyzing Exp 3 (Sweep: Time & Freq) ===")
    r, s = find_latest_pair(["exp3", "sweep", "freq"])
    if r and s: plot_exp3_analysis(r, s)
    
    print(f"\nAnalysis Completed. Check '{OUTPUT_DIR}'")