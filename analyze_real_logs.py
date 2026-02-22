# analyze_real_logs.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import glob

# データのパス設定 (ユーザー環境に合わせて設定)
DATA_DIR = "external_data/jetson_project/deploy_results"
OUTPUT_DIR = "external_data/jetson_project/analysis_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_and_plot(csv_file, is_gmd):
    filename = os.path.basename(csv_file)
    print(f"\n=== 解析開始: {filename} ===")
    
    try:
        df = pd.read_csv(csv_file)
        
        # 変更箇所1: スケールミスの救済
        # 今回のGMDデータ等の target_force が約2Nになってしまっているため、
        # 解析上、強制的に20Nスケール（10倍）に引き上げて補正する。
        df['target_force'] = df['target_force'] * 10.0
        
    except Exception as e:
        print(f"読み込みエラー: {e}")
        return

    # --- 1. 定量スコア計算 (タイミング誤差 & 力の誤差) ---
    # 変更箇所2: 固定閾値(3.0)から、動的閾値に変更
    # データスケールのミスや曲による振幅の違いを吸収するため、最大値の40%をピーク検出の閾値とする
    threshold_target = max(1.0, df['target_force'].max() * 0.4)
    threshold_force = max(1.0, df['force_N'].max() * 0.4)
    
    # 距離(distance)も少し余裕を持たせる
    peaks_target, _ = find_peaks(df['target_force'], height=threshold_target, distance=10) 
    peaks_force, _ = find_peaks(df['force_N'], height=threshold_force, distance=10)
    
    timing_errors_ms = []
    # --- 先行入力(Lookahead)の解析用リスト ---
    lookahead_times_ms = []

    for pt in peaks_target:
        time_t = df['time'].iloc[pt]
        
        # 【追加解析】モデルの先行入力（早めのバルブ開放）時間を計算
        # ターゲットピークの少し前(例えば0.2秒前)からピークまでの間で、cmd_DFが立ち上がり始めた時間を探す
        search_start_idx = max(0, pt - 20) # 20ステップ(約0.2秒)前を探索開始位置と仮定
        cmd_df_window = df['cmd_DF'].iloc[search_start_idx:pt]
        # cmd_DFが0.1(10%)を超えた最初のタイミングを「動き出し」と定義
        if not cmd_df_window.empty and (cmd_df_window > 0.1).any():
            start_idx = cmd_df_window[cmd_df_window > 0.1].index[0]
            time_cmd_start = df['time'].iloc[start_idx]
            lookahead_ms = (time_t - time_cmd_start) * 1000.0
            if lookahead_ms > 0:
                lookahead_times_ms.append(lookahead_ms)

        if len(peaks_force) > 0:
            # 最も時間的に近い実機の力(force_N)のピークを探す
            closest_pf = min(peaks_force, key=lambda pf: abs(df['time'].iloc[pf] - time_t))
            time_f = df['time'].iloc[closest_pf]
            error_ms = (time_f - time_t) * 1000.0  # 秒をミリ秒に変換
            
            # 極端な外れ値は除外 (±150ms以内を有効とする)
            if abs(error_ms) < 150: 
                timing_errors_ms.append(error_ms)
                
    mean_delay = np.mean(timing_errors_ms) if timing_errors_ms else 0.0
    std_delay = np.std(timing_errors_ms) if timing_errors_ms else 0.0
    mae_force = np.mean(np.abs(df['target_force'] - df['force_N']))
    mean_lookahead = np.mean(lookahead_times_ms) if lookahead_times_ms else 0.0

    print(f"🎯 [スコア] Force MAE: {mae_force:.2f} N (補正後 target_force 基準)")
    print(f"⏱️ [スコア] Timing Delay: {mean_delay:.1f} ms (± {std_delay:.1f} ms) / Hit count: {len(timing_errors_ms)}")
    print(f"🧠 [創発] Mean Lookahead (先行入力): {mean_lookahead:.1f} ms 先行してバルブを開放")

    # --- 2. 論文用グラフ作成 ---
    plt.rcParams["font.family"] = "sans-serif"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # 上段: Target vs Real Force
    ax1.plot(df['time'], df['target_force'], label='Target Force (Scaled)', linestyle='--', color='gray', alpha=0.8)
    ax1.plot(df['time'], df['force_N'], label='Real Force', color='blue', linewidth=1.5)
    ax1.set_ylabel('Force [N]', fontsize=12)
    ax1.set_title(f'Tracking Performance ({filename.split("_modelB")[0]})', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 下段: 剛性変調 (Co-contraction / Stiffness)
    stiffness_index = df['cmd_DF'] + df['cmd_F']
    ax2.plot(df['time'], df['cmd_DF'], label='Flexor (DF)', color='red', alpha=0.6)
    ax2.plot(df['time'], df['cmd_F'], label='Extensor (F)', color='green', alpha=0.6)
    
    if is_gmd:
        # 剛性インデックスを強調して表示（エリアを塗りつぶして視覚的にわかりやすく）
        ax2.fill_between(df['time'], 0, stiffness_index, color='black', alpha=0.1, label='Stiffness Index Area')
        ax2.plot(df['time'], stiffness_index, label='Stiffness Index (DF+F)', color='black', linewidth=1.5)
        ax2.set_title('Autonomous Stiffness Modulation', fontsize=12)
    else:
        ax2.set_title('Muscle Pressures', fontsize=12)

    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('Command Pressure [MPa]', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    # 画像として保存
    out_name = os.path.join(OUTPUT_DIR, filename.replace('.csv', '_analysis.png'))
    plt.savefig(out_name, dpi=300)
    plt.close()
    print(f"✅ グラフを保存しました: {out_name}")

if __name__ == "__main__":
    print(f"データの読み込み元: {DATA_DIR}")
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    if not csv_files:
        print("CSVファイルが見つかりません。パスを確認してください。")
    else:
        for file in sorted(csv_files):
            is_gmd = 'gmd' in os.path.basename(file)
            analyze_and_plot(file, is_gmd)
            
        print(f"\n🎉 全ての解析が完了しました！ 結果は {OUTPUT_DIR} を確認してください。")