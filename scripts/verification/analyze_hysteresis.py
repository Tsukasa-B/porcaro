import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def analyze_differential_hysteresis():
    # 1. データ検索
    files = glob.glob('**/01-11-hysteresis.csv', recursive=True)
    if not files:
        files = glob.glob('**/*hysteresis.csv', recursive=True)
    
    if not files:
        print("Error: File not found.")
        return

    target_file = files[0]
    print(f"Analyzing: {target_file}")
    df = pd.read_csv(target_file)
    
    # 2. 差圧 (Differential Pressure) の計算
    # 拮抗駆動の基本: 角度は (P_DF - P_F) に相関するはず
    # ※列名は実際のCSVに合わせて確認・調整してください
    try:
        df['p_diff'] = df['meas_pres_DF'] - df['meas_pres_F']
    except KeyError:
        print("Error: meas_pres_DF or meas_pres_F column not found.")
        print("Columns:", df.columns)
        return

    # 3. 前処理 (スムージング)
    window = 10
    df['p_input'] = df['p_diff'].rolling(window=window, center=True).mean()
    df['angle'] = df['meas_ang_wrist'].rolling(window=window, center=True).mean()
    df = df.dropna()

    # 4. Loading / Unloading 分割
    # 差圧が増えているか減っているかで判定
    df['dp'] = df['p_input'].diff()
    threshold = 0.0001
    
    df_load = df[df['dp'] > threshold]   # 差圧上昇 (手首UP方向へ力が増加)
    df_unload = df[df['dp'] < -threshold] # 差圧減少 (手首DOWN方向へ力が増加)

    print(f"Data Points - Loading: {len(df_load)}, Unloading: {len(df_unload)}")

    if len(df_load) == 0 or len(df_unload) == 0:
        print("Error: Could not split data based on differential pressure.")
        return

    # 5. フィッティング
    degree = 3
    coeffs_load = np.polyfit(df_load['p_input'], df_load['angle'], degree)
    coeffs_unload = np.polyfit(df_unload['p_input'], df_unload['angle'], degree)

    # 6. 結果出力
    print("\n" + "="*60)
    print(" DIFFERENTIAL HYSTERESIS MODEL (Antagonistic)")
    print("="*60)
    print("Model: Angle = f(P_DF - P_F)")
    print("-" * 40)
    print("【Loading (Diff Pressure Increasing)】")
    print(f" Coeffs: {list(coeffs_load)}")
    print("-" * 40)
    print("【Unloading (Diff Pressure Decreasing)】")
    print(f" Coeffs: {list(coeffs_unload)}")
    print("="*60 + "\n")

    # 7. プロット
    plt.figure(figsize=(10, 6))
    
    # 横軸を差圧にしてプロット
    plt.scatter(df['p_diff'], df['meas_ang_wrist'], s=1, c='gray', alpha=0.3, label='Raw Data')
    
    # モデル線
    x_line = np.linspace(df['p_input'].min(), df['p_input'].max(), 100)
    y_load = np.polyval(coeffs_load, x_line)
    y_unload = np.polyval(coeffs_unload, x_line)
    
    plt.plot(x_line, y_load, 'r-', linewidth=2, label='Model (Loading)')
    plt.plot(x_line, y_unload, 'b-', linewidth=2, label='Model (Unloading)')
    
    plt.xlabel("Differential Pressure (P_DF - P_F) [MPa]")
    plt.ylabel("Wrist Angle [deg]")
    plt.title("Antagonistic Hysteresis Analysis")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("hysteresis_differential_check.png")
    print("Graph saved: hysteresis_differential_check.png")

if __name__ == "__main__":
    analyze_differential_hysteresis()