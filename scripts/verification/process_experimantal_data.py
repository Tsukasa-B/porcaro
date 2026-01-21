import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re

# ==========================================
# 設定
# ==========================================
DATA_DIR = "external_data/jetson_project"  # データがあるディレクトリ
OUTPUT_DIR = "analysis_results"            # 解析結果（画像）の保存先

def setup_plot_style():
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['lines.linewidth'] = 2

def load_csv(filepath):
    """CSVを読み込み、タイムスタンプを相対時間に変換する"""
    try:
        df = pd.read_csv(filepath)
        if 'timestamp' in df.columns and not df.empty:
            df['time_rel'] = df['timestamp'] - df['timestamp'].iloc[0]
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load {filepath}: {e}")
        return None

# ==========================================
# Exp 1: Step Response Analysis
# ==========================================
def analyze_exp1(files):
    if not files: return
    print(f"\n--- Analyzing Exp 1 (Step Response): {len(files)} files ---")
    
    for fp in files:
        df = load_csv(fp)
        if df is None: continue
        
        filename = os.path.basename(fp)
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # 1. Pressure F
        axes[0].plot(df['time_rel'], df['cmd_F'], 'r--', label='Cmd F')
        axes[0].plot(df['time_rel'], df['meas_pres_F'], 'r', label='Meas F', alpha=0.7)
        axes[0].set_ylabel('Pressure [MPa]')
        axes[0].set_title(f'[Exp1] Step Response: {filename}')
        axes[0].legend()
        
        # 2. Wrist Angle
        axes[1].plot(df['time_rel'], df['meas_ang_wrist'], 'k', label='Wrist Angle')
        axes[1].set_ylabel('Angle [deg]')
        axes[1].legend()

        # 3. Pressure DF (Check for leaks/interference)
        axes[2].plot(df['time_rel'], df['meas_pres_DF'], 'g', label='Meas DF (Should be 0)')
        axes[2].set_ylabel('Pressure [MPa]')
        axes[2].set_xlabel('Time [s]')
        axes[2].legend()
        
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"Exp1_Step_{filename[:-4]}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

# ==========================================
# Exp 2: Hysteresis Analysis (Multi-Freq)
# ==========================================
def analyze_exp2(files):
    if not files: return
    print(f"\n--- Analyzing Exp 2 (Hysteresis): {len(files)} files ---")
    
    # 周波数ごとにデータを整理
    freq_map = {}
    for fp in files:
        # ファイル名から周波数を抽出 (例: ...0.5Hz...)
        match = re.search(r'([\d\.]+)Hz', fp)
        freq = match.group(1) if match else "Unknown"
        freq_map[freq] = fp

    # プロット準備: 差圧 vs 角度
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.jet(np.linspace(0, 1, len(freq_map)))
    
    for (freq, fp), color in zip(sorted(freq_map.items()), colors):
        df = load_csv(fp)
        if df is None: continue
        
        # 差圧 (Diff Pressure) = DF (Extensor) - F (Flexor)
        # ※ 実験計画通りなら DFが伸展(正?)、Fが屈曲(負?) だが、符号はデータを見て調整
        diff_pressure = df['meas_pres_DF'] - df['meas_pres_F']
        
        # 最初の数秒は過渡応答として除外しても良いが、今回は全プロット
        ax.plot(diff_pressure, df['meas_ang_wrist'], label=f'{freq} Hz', color=color, alpha=0.6, linewidth=1.5)

    ax.set_title('[Exp2] Multi-Axis Hysteresis Loop (Diff Pressure vs Wrist Angle)')
    ax.set_xlabel('Differential Pressure (P_DF - P_F) [MPa]')
    ax.set_ylabel('Wrist Angle [deg]')
    ax.legend()
    
    save_path = os.path.join(OUTPUT_DIR, "Exp2_Hysteresis_Comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

# ==========================================
# Exp 3: Slack Identification
# ==========================================
def analyze_exp3(files):
    if not files: return
    print(f"\n--- Analyzing Exp 3 (Slack ID): {len(files)} files ---")
    
    for fp in files:
        df = load_csv(fp)
        if df is None: continue
        filename = os.path.basename(fp)
        
        # ターゲット筋肉の判定
        target_muscle = "Unknown"
        if "pamdf" in filename: target_muscle = "DF (Extensor)"
        elif "pamf" in filename: target_muscle = "F (Flexor)"
        elif "pamg" in filename: target_muscle = "G (Grip)"
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.set_title(f'[Exp3] Slack Identification: {target_muscle}\n{filename}')
        
        # X軸: 該当筋肉の圧力
        if "pamdf" in filename:
            x_data = df['meas_pres_DF']
            y_data = df['meas_ang_wrist']
            input_label = "Pressure DF"
        elif "pamf" in filename:
            x_data = df['meas_pres_F']
            y_data = df['meas_ang_wrist']
            input_label = "Pressure F"
        elif "pamg" in filename:
            x_data = df['meas_pres_G']
            y_data = df['meas_ang_hand'] # GripはHand Angleを見る
            input_label = "Pressure G"
        else:
            continue

        color = 'tab:red'
        ax1.set_xlabel(f'{input_label} [MPa]')
        ax1.set_ylabel('Angle [deg]', color=color)
        ax1.plot(x_data, y_data, color=color, label='Angle')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # 角度の微分（変化率）を表示して動き出しを見やすくする
        ax2 = ax1.twinx()
        color = 'tab:blue'
        # 簡易微分 (ノイズ除去のためRolling平均後にDiff)
        angle_diff = y_data.rolling(10).mean().diff().abs()
        ax2.set_ylabel('Angle Change Rate (abs)', color=color)
        ax2.plot(x_data, angle_diff, color=color, alpha=0.3, label='|dAngle/dP|')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"Exp3_Slack_{filename[:-4]}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

# ==========================================
# Exp 4: Validation Sequence
# ==========================================
def analyze_exp4(files):
    if not files: return
    print(f"\n--- Analyzing Exp 4 (Validation): {len(files)} files ---")
    
    for fp in files:
        df = load_csv(fp)
        if df is None: continue
        filename = os.path.basename(fp)

        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        
        # 1. Wrist Angle
        axes[0].plot(df['time_rel'], df['meas_ang_wrist'], 'k', label='Meas Wrist')
        axes[0].set_ylabel('Angle [deg]')
        axes[0].set_title(f'[Exp4] Validation Sequence: {filename}')
        axes[0].legend()
        
        # 2. Hand Angle
        axes[1].plot(df['time_rel'], df['meas_ang_hand'], 'k', label='Meas Hand')
        axes[1].set_ylabel('Angle [deg]')
        axes[1].legend()

        # 3. Pressures (F & DF)
        axes[2].plot(df['time_rel'], df['cmd_DF'], 'g--', label='Cmd DF', alpha=0.5)
        axes[2].plot(df['time_rel'], df['meas_pres_DF'], 'g', label='Meas DF')
        axes[2].plot(df['time_rel'], df['cmd_F'], 'r--', label='Cmd F', alpha=0.5)
        axes[2].plot(df['time_rel'], df['meas_pres_F'], 'r', label='Meas F')
        axes[2].set_ylabel('Wrist PAMs [MPa]')
        axes[2].legend(loc='upper right')
        
        # 4. Pressure (Grip)
        axes[3].plot(df['time_rel'], df['cmd_G'], 'b--', label='Cmd G', alpha=0.5)
        axes[3].plot(df['time_rel'], df['meas_pres_G'], 'b', label='Meas G')
        axes[3].set_ylabel('Grip PAM [MPa]')
        axes[3].set_xlabel('Time [s]')
        axes[3].legend(loc='upper right')

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"Exp4_Validation_{filename[:-4]}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

# ==========================================
# Main Process
# ==========================================
def main():
    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Data directory not found: {DATA_DIR}")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_plot_style()

    # ファイル検索
    all_csvs = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"Found {len(all_csvs)} CSV files in {DATA_DIR}")

    # 分類
    files_exp1 = [f for f in all_csvs if "exp1" in os.path.basename(f)]
    files_exp2 = [f for f in all_csvs if "exp2" in os.path.basename(f)]
    files_exp3 = [f for f in all_csvs if "exp3" in os.path.basename(f)]
    files_exp4 = [f for f in all_csvs if "exp4" in os.path.basename(f)]

    # 解析実行
    analyze_exp1(files_exp1)
    analyze_exp2(files_exp2)
    analyze_exp3(files_exp3)
    analyze_exp4(files_exp4)

    print(f"\nAll analyses completed. Check the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()