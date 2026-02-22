"""
IROS 2026 Figure 1 (Integrated): Model Validation
Hysteresis, Step Response, and Frequency Sweep Response
(右軸圧力スケール追加 ＆ レイアウト統合版)
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
        'figure.autolayout': False, # GridSpecを使うためFalseにする
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

    if not real_cands:
        return None, None
    real_cands.sort(key=lambda x: (0 if "data_" in os.path.basename(x) else 1, -os.path.getctime(x)))
    return real_cands[0], max(sim_cands, key=os.path.getctime) if sim_cands else None

def load_and_sync_by_cmd(real_path, sim_path):
    df_r = pd.read_csv(real_path)
    df_s = pd.read_csv(sim_path) if sim_path else None

    t_col_r = 'time' if 'time' in df_r.columns else 'timestamp_pc'
    
    if 'cmd_DF' in df_r.columns and (df_r['cmd_DF'] > 0.05).any():
        t0_r = df_r.loc[df_r['cmd_DF'] > 0.05, t_col_r].iloc[0]
    else:
        t0_r = df_r[t_col_r].iloc[0]
    df_r['t_sync'] = df_r[t_col_r] - t0_r

    if df_s is not None:
        t_col_s = 'time'
        if 'cmd_DF' in df_s.columns and (df_s['cmd_DF'] > 0.05).any():
            t0_s = df_s.loc[df_s['cmd_DF'] > 0.05, t_col_s].iloc[0]
        else:
            t0_s = df_s[t_col_s].iloc[0]
        df_s['t_sync'] = df_s[t_col_s] - t0_s

    return df_r, df_s

def plot_integrated_figure_1():
    set_ieee_style()
    OUTPUT_DIR = "iros_figures"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    C_REAL = '#D62728' # 赤系 (Real)
    C_SIM = '#1F77B4'  # 青系 (Sim)
    C_CMD = 'gray'     # グレー (Command)

    # 全体のFigure作成 (IEEE 2カラム幅に合わせる: 幅7.16インチ, 高さ約5.5インチ)
    fig = plt.figure(figsize=(7.16, 5.5))
    
    # 2行2列のGridSpecを作成 (下段は2列を結合する)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.4, wspace=0.3)
    ax_a = fig.add_subplot(gs[0, 0]) # 左上: ヒステリシス
    ax_b = fig.add_subplot(gs[0, 1]) # 右上: ステップ応答
    ax_c = fig.add_subplot(gs[1, :]) # 下段ぶち抜き: 周波数スイープ

    # ===================================================
    # (a) ヒステリシスループ (Exp 1)
    # ===================================================
    r_path1, s_path1 = find_latest_pair(["exp1", "static"])
    if r_path1:
        df_r1, df_s1 = load_and_sync_by_cmd(r_path1, s_path1)
        ax_a.plot(df_r1['meas_pres_DF'], df_r1['angle_deg'], color=C_REAL, label='Real Robot', zorder=2)
        if df_s1 is not None:
            ax_a.plot(df_s1['sim_pres_DF'], df_s1['sim_angle_deg'], color=C_SIM, linestyle='--', label='Sim (Model B)', zorder=3)

    ax_a.set_xlabel('Pressure $P_{DF}$ (MPa)')
    ax_a.set_ylabel('Joint Angle (deg)')
    ax_a.set_title('(a) Static Hysteresis')
    ax_a.grid(True)
    ax_a.legend(loc='lower right')

    # ===================================================
    # (b) ステップ応答 (Exp 2) - 第2Y軸（圧力）追加
    # ===================================================
    r_path2, s_path2 = find_latest_pair(["exp2", "step"])
    if r_path2:
        df_r2, df_s2 = load_and_sync_by_cmd(r_path2, s_path2)
        
        ax_b.plot(df_r2['t_sync'], df_r2['angle_deg'], color=C_REAL, label='Real', zorder=2)
        if df_s2 is not None:
            ax_b.plot(df_s2['t_sync'], df_s2['sim_angle_deg'], color=C_SIM, linestyle='--', label='Sim', zorder=3)
        
        # --- 第2Y軸（圧力）の作成 ---
        ax_b_cmd = ax_b.twinx()
        ax_b_cmd.plot(df_r2['t_sync'], df_r2['cmd_DF'], color=C_CMD, linestyle=':', alpha=0.7, label='Command', zorder=1)
        ax_b_cmd.set_ylabel('Command (MPa)', color=C_CMD)
        ax_b_cmd.set_ylim(-0.066, 0.65)
        ax_b_cmd.tick_params(axis='y', labelcolor=C_CMD)

        ax_b.set_xlim(-0.2, 5.0)
        ax_b.set_ylim(df_r2['angle_deg'].min() - 5, df_r2['angle_deg'].max() + 10)

        # 凡例を統合して表示（左軸の凡例だけでスッキリさせるか、両方乗せるか）
        lines_b, labels_b = ax_b.get_legend_handles_labels()
        lines_cmd, labels_cmd = ax_b_cmd.get_legend_handles_labels()
        ax_b.legend(lines_b + lines_cmd, labels_b + labels_cmd, loc='lower right')

    ax_b.set_xlabel('Time (s)')
    ax_b.set_ylabel('Joint Angle (deg)')
    ax_b.set_title('(b) Step Response')
    ax_b.grid(True)

    # ===================================================
    # (c) 周波数スイープ (Exp 3) - 第2Y軸（圧力）追加
    # ===================================================
    r_path3, s_path3 = find_latest_pair(["exp3", "sweep"])
    if r_path3:
        df_r3, df_s3 = load_and_sync_by_cmd(r_path3, s_path3)

        ax_c.plot(df_r3['t_sync'], df_r3['angle_deg'], color=C_REAL, label='Real Robot', zorder=2)
        if df_s3 is not None and 'sim_angle_deg' in df_s3.columns:
            ax_c.plot(df_s3['t_sync'], df_s3['sim_angle_deg'], color=C_SIM, linestyle='--', label='Sim (Model B)', zorder=3)

        # --- 第2Y軸（圧力）の作成 ---
        ax_c_cmd = ax_c.twinx()
        if 'cmd_DF' in df_r3.columns:
            ax_c_cmd.plot(df_r3['t_sync'], df_r3['cmd_DF'], color=C_CMD, linestyle=':', alpha=0.5, label='Command Input', zorder=1)
        ax_c_cmd.set_ylabel('Command (MPa)', color=C_CMD)
        ax_c_cmd.set_ylim(-0.085, 0.65)
        ax_c_cmd.tick_params(axis='y', labelcolor=C_CMD)

        max_t = df_r3['t_sync'].max()
        ax_c.set_xlim(-0.5, min(20.0, max_t)) 

        # 注釈
        ax_c.text(ax_c.get_xlim()[1]*0.1, ax_c.get_ylim()[1]*0.85, 'Low Freq:\nWell Matched', color='green', fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax_c.text(ax_c.get_xlim()[1]*0.85, ax_c.get_ylim()[1]*0.85, 'High Freq:\nPhase Shift & Attenuation', color='black', fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # 凡例
        lines_c, labels_c = ax_c.get_legend_handles_labels()
        lines_cmd_c, labels_cmd_c = ax_c_cmd.get_legend_handles_labels()
        ax_c.legend(lines_c + lines_cmd_c, labels_c + labels_cmd_c, loc='upper right', framealpha=0.9)

    ax_c.set_xlabel('Time (s)')
    ax_c.set_ylabel('Joint Angle (deg)')
    ax_c.set_title('(c) Frequency Sweep Response (Reality Gap at High Frequencies)')
    ax_c.grid(True)

    # 出力
    out_svg = os.path.join(OUTPUT_DIR, "Fig1_Integrated_ModelValidation.svg")
    out_png = os.path.join(OUTPUT_DIR, "Fig1_Integrated_ModelValidation.png")
    fig.savefig(out_svg, format='svg', bbox_inches='tight')
    fig.savefig(out_png, format='png', bbox_inches='tight')
    plt.close()
    
    print(f"✅ Integrated Figure 1 saved: {out_png}")

if __name__ == "__main__":
    plot_integrated_figure_1()