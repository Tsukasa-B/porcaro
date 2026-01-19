# scripts/verification/analyze_hysteresis_width.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# データパス設定
csv_path = "external_data/jetson_project/01-11-hysteresis.csv"

def analyze_hysteresis(csv_file):
    if not os.path.exists(csv_file):
        print(f"Error: File not found at {csv_file}")
        return

    df = pd.read_csv(csv_file)
    
    # 時系列データの確認 (等間隔サンプリングを仮定しない)
    time = df['timestamp'].values
    time = time - time[0]
    
    # 背屈筋 (DF) を代表として分析
    cmd = df['cmd_DF'].values
    meas = df['meas_pres_DF'].values

    # ヒステリシス幅の推定
    # 簡易手法: 同じ指令値付近での計測値の最大差分を取得
    # 0.1~0.6MPaの範囲でビン分割して差分を計算
    bins = np.linspace(0.1, 0.6, 50)
    widths = []
    
    for i in range(len(bins)-1):
        # ビン内のデータマスク
        mask = (cmd >= bins[i]) & (cmd < bins[i+1])
        if np.sum(mask) > 10:
            val_in_bin = meas[mask]
            # 昇圧/降圧が混在していると仮定し、最大値-最小値をそのビンでの幅とする
            w = np.max(val_in_bin) - np.min(val_in_bin)
            widths.append(w)
    
    estimated_width = np.mean(widths) if widths else 0.0
    print(f"Estimated Hysteresis Width (Mean): {estimated_width:.4f} MPa")

    # プロット
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(time, cmd, label='Command', alpha=0.7)
    plt.plot(time, meas, label='Measured', alpha=0.7)
    plt.title("Time Series")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.scatter(cmd, meas, s=1, alpha=0.5, label='Real Data')
    plt.title(f"Hysteresis Loop (Width ~ {estimated_width:.3f})")
    plt.xlabel("Command Pressure")
    plt.ylabel("Measured Pressure")
    plt.grid()
    
    output_img = csv_file.replace('.csv', '_analysis.png')
    plt.savefig(output_img)
    print(f"Analysis plot saved to {output_img}")

if __name__ == "__main__":
    analyze_hysteresis(csv_path)