# generate_iros_figures.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import glob

# --- 設定 ---
DATA_DIR = "external_data/jetson_project/deploy_results"
OUTPUT_DIR = "external_data/jetson_project/analysis_results/figures_iros"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 【重要】実際のCSVの列名に合わせて変更してください
COL_PRESS_DF = 'meas_pres_DF' # 手首 屈筋の実際の圧力
COL_PRESS_F  = 'meas_pres_F'  # 手首 伸筋の実際の圧力
COL_PRESS_G  = 'meas_pres_G'  # グリップ (PAM-G) の実際の圧力
COL_POS_G    = 'grip_angle_deg'         # グリップ関節の実際の角度/位置 (なければ 'cmd_G' 等で代用可能)

# IEEE論文用のグラフ設定
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "lines.linewidth": 2.0,
})

def analyze_single_file(csv_file):
    """単一ファイルの定量スコア（Delay, Lookahead等）を計算する"""
    df = pd.read_csv(csv_file)
    is_gmd = 'gmd' in os.path.basename(csv_file)
    
    if is_gmd:
        df['target_force'] = df['target_force'] * 10.0
        
    threshold_target = max(1.0, df['target_force'].max() * 0.4)
    if 'force_N' in df.columns:
        threshold_force = max(1.0, df['force_N'].max() * 0.4)
        peaks_force, _ = find_peaks(df['force_N'], height=threshold_force, distance=10)
    else:
        peaks_force = []
        
    peaks_target, _ = find_peaks(df['target_force'], height=threshold_target, distance=10) 
    
    timing_errors_ms = []
    lookahead_times_ms = []

    for pt in peaks_target:
        time_t = df['time'].iloc[pt]
        
        search_start_idx = max(0, pt - 25)
        # 先行入力はコマンド(cmd_DF)をベースに計算
        if 'cmd_DF' in df.columns:
            cmd_df_window = df['cmd_DF'].iloc[search_start_idx:pt]
            if not cmd_df_window.empty and (cmd_df_window > 0.1).any():
                start_idx = cmd_df_window[cmd_df_window > 0.1].index[0]
                time_cmd_start = df['time'].iloc[start_idx]
                lookahead_ms = (time_t - time_cmd_start) * 1000.0
                if lookahead_ms > 0:
                    lookahead_times_ms.append(lookahead_ms)

        if len(peaks_force) > 0:
            closest_pf = min(peaks_force, key=lambda pf: abs(df['time'].iloc[pf] - time_t))
            time_f = df['time'].iloc[closest_pf]
            error_ms = (time_f - time_t) * 1000.0
            if abs(error_ms) < 150: 
                timing_errors_ms.append(error_ms)
                
    return {
        'mean_delay': np.mean(timing_errors_ms) if timing_errors_ms else 0,
        'mean_lookahead': np.mean(lookahead_times_ms) if lookahead_times_ms else 0,
        'df': df,
        'is_gmd': is_gmd
    }

def generate_fig4_adaptive_anticipation(all_results):
    """Fig 4: BPMに応じたLookaheadの増加とDelayの推移"""
    print("\n[Fig. 4] 身体的知能の定量評価グラフを作成中...")
    
    bpms = []
    lookaheads = []
    delays = []
    
    for filename, res in all_results.items():
        if 'gmd' in filename:
            bpm_str = filename.split('_bpm')[1].split('_')[0]
            bpms.append(int(bpm_str))
            lookaheads.append(res['mean_lookahead'])
            delays.append(res['mean_delay'])
            
    if not bpms:
        print("警告: Fig. 4 用のGMDデータが見つかりません。")
        return

    sorted_indices = np.argsort(bpms)
    bpms = np.array(bpms)[sorted_indices]
    lookaheads = np.array(lookaheads)[sorted_indices]
    delays = np.array(delays)[sorted_indices]
    
    fig, ax1 = plt.subplots(figsize=(7, 5))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Groove MIDI Tempo [BPM]')
    ax1.set_ylabel('Anticipatory Lookahead [ms]', color=color1)
    bars = ax1.bar([str(b) for b in bpms], lookaheads, color=color1, alpha=0.7, width=0.5, label='Lookahead Time')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, max(lookaheads) * 1.5)
    
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}', ha='center', va='bottom', color=color1, fontsize=10)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Tracking Delay [ms]', color=color2)
    line = ax2.plot([str(b) for b in bpms], delays, color=color2, marker='o', markersize=8, linewidth=2.5, label='Timing Delay')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, max(max(delays) * 1.5, 60))

    plt.title('Adaptive Lookahead vs. Tracking Delay')
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig4_Adaptive_Anticipation.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig4_Adaptive_Anticipation.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_fig5_single_vs_double(all_results):
    """Fig 5: BPM160でのSingleとDoubleの波形比較 (Grip追加版)"""
    print("[Fig. 5] Single vs Double ダイナミクス比較グラフを作成中...")
    
    df_single = None
    df_double = None
    
    for filename, res in all_results.items():
        if 'test_single8_bpm160' in filename:
            df_single = res['df']
        elif 'test_double_bpm160' in filename:
            df_double = res['df']
            
    if df_single is None or df_double is None:
        print("警告: Fig. 5 用のファイル(bpm160)が見つかりません。")
        return

    fig, axes = plt.subplots(3, 2, figsize=(12, 8.5), sharex=False)
    
    t_start = 2.0
    t_end = 3.0
    
    for col, (df, title) in enumerate([(df_single, 'Single Stroke (160 BPM)'), (df_double, 'Double Stroke (160 BPM)')]):
        mask = (df['time'] >= t_start) & (df['time'] <= t_end)
        df_plot = df[mask]
        t = df_plot['time'] - t_start
        
        # --- 上段: Force ---
        axes[0, col].plot(t, df_plot['target_force'], '--', color='gray', label='Target')
        axes[0, col].plot(t, df_plot['force_N'], color='blue', label='Real Force')
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel('Force [N]')
        if col == 0: axes[0, col].legend(loc='upper right')
        axes[0, col].grid(True, alpha=0.3)
        
        # --- 中段: Wrist Pressures ---
        if COL_PRESS_DF in df_plot.columns and COL_PRESS_F in df_plot.columns:
            axes[1, col].plot(t, df_plot[COL_PRESS_DF], color='red', label='Measured DF')
            axes[1, col].plot(t, df_plot[COL_PRESS_F], color='green', label='Measured F')
            stiffness = df_plot[COL_PRESS_DF] + df_plot[COL_PRESS_F]
            axes[1, col].fill_between(t, 0, stiffness, color='black', alpha=0.1, label='Wrist Stiffness')
            axes[1, col].set_ylabel('Wrist [MPa]')
            axes[1, col].set_ylim(0, df_plot[[COL_PRESS_DF, COL_PRESS_F]].max().max() * 1.5)
            if col == 0: axes[1, col].legend(loc='upper right')
        axes[1, col].grid(True, alpha=0.3)

        # --- 下段: Grip ---
        if COL_PRESS_G in df_plot.columns:
            axes[2, col].plot(t, df_plot[COL_PRESS_G], color='darkorange', label='Measured Grip (PAM-G)')
            axes[2, col].set_ylabel('Grip [MPa]', color='darkorange')
            axes[2, col].tick_params(axis='y', labelcolor='darkorange')
            
            if COL_POS_G in df_plot.columns:
                ax2_right = axes[2, col].twinx()
                ax2_right.plot(t, df_plot[COL_POS_G], color='purple', linestyle='-.', label='Grip Joint Angle')
                if col == 1: 
                    ax2_right.set_ylabel('Angle [rad]', color='purple')
                ax2_right.tick_params(axis='y', labelcolor='purple')
                if col == 0:
                    lines, labels = axes[2, col].get_legend_handles_labels()
                    lines2, labels2 = ax2_right.get_legend_handles_labels()
                    axes[2, col].legend(lines + lines2, labels + labels2, loc='upper right')
            else:
                if col == 0: axes[2, col].legend(loc='upper right')

        axes[2, col].set_xlabel('Time [s]')
        axes[2, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig5_Dynamics_Comparison_with_Grip.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig5_Dynamics_Comparison_with_Grip.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_fig6_gmd_stiffness(all_results):
    """Fig 6: GMD Extremeでの自律的剛性変調 (Grip追加版)"""
    print("[Fig. 6] 未知のグルーヴへの適応(GMD)グラフを作成中...")
    
    df_gmd = None
    for filename, res in all_results.items():
        if 'gmd_04_extreme_bpm170' in filename:
            df_gmd = res['df']
            break
            
    if df_gmd is None:
        print("警告: Fig. 6 用のファイル(gmd_04_extreme)が見つかりません。")
        return

    t_start = 2.0
    t_end = 5.5
    mask = (df_gmd['time'] >= t_start) & (df_gmd['time'] <= t_end)
    df_plot = df_gmd[mask]
    t = df_plot['time'] - t_start

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True, gridspec_kw={'height_ratios': [1, 1.2, 1.2]})
    
    # --- 上段: Target vs Real ---
    ax1.plot(t, df_plot['target_force'], '--', color='gray', label='Target Force', alpha=0.8)
    ax1.plot(t, df_plot['force_N'], color='blue', label='Real Force')
    ax1.set_ylabel('Force [N]')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Zero-shot Adaptation (170 BPM): Wrist & Grip Coordination')
    
    # --- 中段: Wrist Stiffness Index ---
    if COL_PRESS_DF in df_plot.columns and COL_PRESS_F in df_plot.columns:
        real_stiffness = df_plot[COL_PRESS_DF] + df_plot[COL_PRESS_F]
        ax2.fill_between(t, 0, real_stiffness, color='purple', alpha=0.2)
        ax2.plot(t, real_stiffness, color='purple', linewidth=2.5, label='Wrist Stiffness (DF+F)')
        ax2.plot(t, df_plot[COL_PRESS_DF], color='red', alpha=0.5, linestyle=':', label='Measured DF')
        ax2.plot(t, df_plot[COL_PRESS_F], color='green', alpha=0.5, linestyle=':', label='Measured F')
        ax2.set_ylabel('Wrist [MPa]')
        ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # --- 下段: Grip Dynamics ---
    if COL_PRESS_G in df_plot.columns:
        ax3.plot(t, df_plot[COL_PRESS_G], color='darkorange', linewidth=2.0, label='Grip Pressure (PAM-G)')
        ax3.set_ylabel('Grip [MPa]', color='darkorange')
        ax3.tick_params(axis='y', labelcolor='darkorange')
        
        if COL_POS_G in df_plot.columns:
            ax3_right = ax3.twinx()
            ax3_right.plot(t, df_plot[COL_POS_G], color='black', linestyle='-.', alpha=0.8, label='Grip Joint Angle')
            ax3_right.set_ylabel('Angle [rad]')
            
            lines, labels = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3_right.get_legend_handles_labels()
            ax3.legend(lines + lines2, labels + labels2, loc='upper right')
        else:
            ax3.legend(loc='upper right')

    ax3.set_xlabel('Time [s]')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig6_GMD_Stiffness_with_Grip.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig6_GMD_Stiffness_with_Grip.png'), dpi=300, bbox_inches='tight')
    plt.close()

# --- ここが欠けていた実行ブロックです ---
if __name__ == "__main__":
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print("CSVファイルが見つかりません。")
        exit()
        
    print(f"--- 解析および論文用画像生成を開始します ---")
    all_results = {}
    for f in sorted(csv_files):
        filename = os.path.basename(f)
        # 解析を実行
        all_results[filename] = analyze_single_file(f)
        
    # 各論文用図表の生成
    generate_fig4_adaptive_anticipation(all_results)
    generate_fig5_single_vs_double(all_results)
    generate_fig6_gmd_stiffness(all_results)
    
    print(f"\n🎉 完了！ 論文用グラフは {OUTPUT_DIR} に保存されました。")