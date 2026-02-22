"""
IROS 2026 Figure 2: Frequency Sweep Response (Time-Series)
原点回帰 ＆ Reality Gap 可視化の決定版
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def set_ieee_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'lines.linewidth': 1.5,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'figure.autolayout': True,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

def find_latest_pair(exp_keywords):
    search_root = "."
    real_cands, sim_cands = [], []
    for root, _, files in os.walk(search_root):
        for f in files:
            if not f.endswith(".csv") or "analysis" in root: continue
            path = os.path.join(root, f)
            if all(k in f for k in exp_keywords):
                if "sim_log" in f: sim_cands.append(path)
                elif "data_" in f or "exp" in f: real_cands.append(path)

    if not real_cands: return None, None
    real_cands.sort(key=lambda x: (0 if "data_" in os.path.basename(x) else 1, -os.path.getctime(x)))
    return real_cands[0], max(sim_cands, key=os.path.getctime) if sim_cands else None

def load_and_sync_by_cmd(real_path, sim_path):
    """ cmd_DF の立ち上がりで時間を完璧に同期する """
    df_r = pd.read_csv(real_path)
    df_s = pd.read_csv(sim_path) if sim_path else None

    # Real同期
    if 'cmd_DF' in df_r.columns and (df_r['cmd_DF'] > 0.05).any():
        t0_r = df_r.loc[df_r['cmd_DF'] > 0.05, 'time'].iloc[0]
    else: t0_r = df_r['time'].iloc[0]
    df_r['t_sync'] = df_r['time'] - t0_r

    # Sim同期
    if df_s is not None:
        if 'cmd_DF' in df_s.columns and (df_s['cmd_DF'] > 0.05).any():
            t0_s = df_s.loc[df_s['cmd_DF'] > 0.05, 'time'].iloc[0]
        else: t0_s = df_s['time'].iloc[0]
        df_s['t_sync'] = df_s['time'] - t0_s

    return df_r, df_s

def plot_figure_2_timeseries():
    set_ieee_style()
    OUTPUT_DIR = "iros_figures"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    C_REAL = '#D62728' # 赤系 (Real)
    C_SIM = '#1F77B4'  # 青系 (Sim)
    C_CMD = 'gray'     # グレー (Command)

    real_path, sim_path = find_latest_pair(["exp3", "sweep"])
    if not real_path:
        print("Exp3 Sweep data not found.")
        return

    df_r, df_s = load_and_sync_by_cmd(real_path, sim_path)

    # 1カラム幅の横長グラフ (幅約3.5インチ〜広めにしたい場合は7インチ)
    # ここでは詳細が見えるように少し横長 (7.16インチ) に設定します
    fig, ax = plt.subplots(figsize=(7.16, 3.0))

    # 指令値(cmd_DF)を背景にプロット（スケールを角度に合わせて見やすくする）
    if 'cmd_DF' in df_r.columns:
        max_angle = df_r['angle_deg'].max()
        ax.plot(df_r['t_sync'], df_r['cmd_DF'] * max_angle, color=C_CMD, linestyle=':', alpha=0.6, label='Command Input', zorder=1)

    # Realの角度プロット
    ax.plot(df_r['t_sync'], df_r['angle_deg'], color=C_REAL, label='Real Robot', zorder=2)

    # Simの角度プロット
    if df_s is not None and 'sim_angle_deg' in df_s.columns:
        ax.plot(df_s['t_sync'], df_s['sim_angle_deg'], color=C_SIM, linestyle='--', label='Sim (Model B)', zorder=3)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Joint Angle (deg)')
    ax.set_title('Fig. 2: Frequency Sweep Response (Reality Gap at High Frequencies)')
    ax.grid(True)
    
    # 凡例は右上（または波形が被らない場所）に
    ax.legend(loc='upper right', framealpha=0.9)

    # X軸のズーム: cmdが開始する t=0 から、スイープが終わる付近まで
    # ※ 実験の長さに合わせて 10.0 などを変更してください。
    max_t = df_r['t_sync'].max()
    ax.set_xlim(-0.5, min(12.0, max_t)) 

    # 注釈を追加（右に行くほどズレることをアピール）
    ax.text(ax.get_xlim()[1]*0.1, ax.get_ylim()[1]*0.85, 'Low Freq:\nWell Matched', color='green', fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax.text(ax.get_xlim()[1]*0.8, ax.get_ylim()[1]*0.85, 'High Freq:\nPhase Shift & Attenuation', color='black', fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    out_svg = os.path.join(OUTPUT_DIR, "Fig2_Sweep_TimeSeries.svg")
    out_png = os.path.join(OUTPUT_DIR, "Fig2_Sweep_TimeSeries.png")
    plt.savefig(out_svg, format='svg')
    plt.savefig(out_png, format='png')
    plt.close()
    
    print(f"✅ Figure 2 (Time-Series) saved: {out_png}")

if __name__ == "__main__":
    plot_figure_2_timeseries()