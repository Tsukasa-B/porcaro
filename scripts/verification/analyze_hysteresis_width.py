import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# CSVファイル名
CSV_FILE = "external_data/jetson_project/01-11-hysteresis.csv"

def calculate_hysteresis_parameters():
    print(f"Loading {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} が見つかりません。")
        return

    # ヒステリシスモードのデータのみ抽出
    # (ファイル内に他のモードが混ざっている場合に備えてフィルタリング)
    if 'mode' in df.columns:
        df = df[df['mode'].astype(str).str.contains('hysteresis')]
    
    # 対象とする筋肉（ここではFlexor: Fをメインに解析）
    target_cmd = 'cmd_F'
    target_meas = 'meas_pres_F'
    
    if target_cmd not in df.columns:
        print(f"Error: {target_cmd} 列が見つかりません。")
        return

    # データ取得
    cmd = df[target_cmd].values
    meas = df[target_meas].values
    time = df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(cmd))

    # ノイズ除去（計測値が見にくい場合用、軽めにかける）
    # meas_smooth = savgol_filter(meas, window_length=11, polyorder=2)
    meas_smooth = meas # 生データで見る場合はこちら

    # --- 上昇（Loading）と下降（Unloading）の分離 ---
    # 差分をとって符号で判定
    diff_cmd = np.diff(cmd)
    # ノイズ対策：ある程度大きな変化だけを見る
    threshold = 0.0001
    
    loading_indices = np.where(diff_cmd > threshold)[0]
    unloading_indices = np.where(diff_cmd < -threshold)[0]
    
    # プロット準備
    plt.figure(figsize=(12, 6))
    
    # 1. Cmd vs Meas プロット (ヒステリシスループ)
    plt.subplot(1, 2, 1)
    plt.scatter(cmd[loading_indices], meas_smooth[loading_indices], s=1, c='r', label='Loading (Increase)')
    plt.scatter(cmd[unloading_indices], meas_smooth[unloading_indices], s=1, c='b', label='Unloading (Decrease)')
    plt.plot([0, 0.6], [0, 0.6], 'k--', alpha=0.5, label='Ideal y=x')
    
    plt.xlabel('Command Pressure (MPa)')
    plt.ylabel('Measured Pressure (MPa)')
    plt.title(f'Hysteresis Loop: {target_cmd} vs {target_meas}')
    plt.legend()
    plt.grid(True)

    # --- ヒステリシス幅の算出 ---
    # 同じCommand値におけるMeasの差分を計算したいが、
    # サンプリング点が一致しないため、Command値を基準にビン(区間)分けして平均を取る
    
    bins = np.linspace(0.1, 0.5, 50) # 0.1MPa〜0.5MPaの間を50分割して解析
    widths = []
    valid_bins = []

    for i in range(len(bins) - 1):
        low = bins[i]
        high = bins[i+1]
        mid = (low + high) / 2
        
        # この区間にあるLoadingデータのMeas平均
        mask_load = (cmd[loading_indices] >= low) & (cmd[loading_indices] < high)
        if not np.any(mask_load): continue
        val_load = np.mean(meas_smooth[loading_indices][mask_load])
        
        # この区間にあるUnloadingデータのMeas平均
        mask_unload = (cmd[unloading_indices] >= low) & (cmd[unloading_indices] < high)
        if not np.any(mask_unload): continue
        val_unload = np.mean(meas_smooth[unloading_indices][mask_unload])
        
        # 幅 = 帰り - 行き
        width = val_unload - val_load
        
        # 幅が負になる（計測ノイズや交差）箇所は除外してもよいが、分布を見るため残す
        widths.append(width)
        valid_bins.append(mid)

    widths = np.array(widths)
    
    # 統計値
    mean_width = np.mean(widths)
    max_width = np.max(widths)
    median_width = np.median(widths)

    print("-" * 50)
    print(f"【解析結果】 {target_cmd} -> {target_meas}")
    print(f"  Mean Width (平均幅): {mean_width:.4f} MPa")
    print(f"  Median Width (中央値): {median_width:.4f} MPa")
    print(f"  Max Width (最大幅):   {max_width:.4f} MPa")
    print("-" * 50)
    print(f"★ 推奨パラメータ (hysteresis_width): {median_width:.4f} 〜 {mean_width:.4f}")
    print("-" * 50)

    # 2. Command vs Hysteresis Width
    plt.subplot(1, 2, 2)
    plt.plot(valid_bins, widths, 'g-o')
    plt.axhline(mean_width, color='r', linestyle='--', label=f'Mean: {mean_width:.3f}')
    plt.axhline(median_width, color='b', linestyle=':', label=f'Median: {median_width:.3f}')
    
    plt.xlabel('Command Pressure (MPa)')
    plt.ylabel('Hysteresis Width (Diff) [MPa]')
    plt.title('Hysteresis Width Distribution')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('hysteresis_analysis_result.png')
    print("グラフを 'hysteresis_analysis_result.png' に保存しました。")
    plt.show()

if __name__ == "__main__":
    calculate_hysteresis_parameters()