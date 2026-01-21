# analyze_slack_smoothness.py
# 目的: 実機データの「動き出し」部分を解析し、Slack OffsetとSmoothnessを同定する

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse

def softplus_model(x, y_base, k, x0, alpha):
    """
    Softplusモデル: カドが丸まった折れ線グラフ
    x: 入力（圧力）
    y_base: 初期角度
    k: 傾き（感度）
    x0: 動き出しの閾値 (Slack Offset)
    alpha: 滑らかさ (Smoothness)
    """
    # 数値安定性のための処理
    z = alpha * (x - x0)
    # log(1 + exp(z)) の安定実装
    sp = np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)
    
    # モデル: y = y_base + (k/alpha) * softplus(alpha * (x - x0))
    # alpha -> inf で ReLU (x > x0 で傾き k) になる
    return y_base + (k / alpha) * sp

def analyze_slack(csv_file, pressure_col='meas_pres_DF', angle_col='meas_ang_wrist'):
    print(f"Loading: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # データの前処理: 圧力が上昇している区間（往路）のみ抽出
    peak_idx = df[pressure_col].idxmax()
    df_ramp = df.iloc[:peak_idx]
    
    # ノイズ除去（負の圧力など）
    df_ramp = df_ramp[df_ramp[pressure_col] > 0.0]
    
    X_data = df_ramp[pressure_col].values
    Y_data = df_ramp[angle_col].values
    
    # 初期推定値 (Initial Guess)
    # y_base: 最小角度, k: 全体の傾き, x0: 0.15MPa付近, alpha: 30付近
    p0 = [np.min(Y_data), (np.max(Y_data)-np.min(Y_data))/0.6, 0.15, 30.0]
    
    try:
        # フィッティング実行
        popt, pcov = curve_fit(softplus_model, X_data, Y_data, p0=p0, maxfev=10000)
        y_base_opt, k_opt, x0_opt, alpha_opt = popt
        
        print("\n" + "="*40)
        print("   解析結果 (Optimization Results)")
        print("="*40)
        print(f"1. Slack Offset (x0) : {x0_opt:.4f} MPa")
        print(f"   -> これ以下の圧力では力が伝達されません")
        print(f"2. Smoothness (alpha): {alpha_opt:.4f} (1/MPa basis)")
        print(f"   -> Isaac Lab推奨値 (収縮率換算) : {alpha_opt * 3.0:.1f} ~ {alpha_opt * 4.0:.1f}")
        print("="*40 + "\n")
        
        # プロット作成
        plt.figure(figsize=(10, 6))
        plt.scatter(X_data, Y_data, s=5, alpha=0.3, color='gray', label='Real Data')
        
        x_pred = np.linspace(min(X_data), max(X_data), 500)
        y_pred = softplus_model(x_pred, *popt)
        
        plt.plot(x_pred, y_pred, 'r-', linewidth=2, label=f'Fit (alpha={alpha_opt:.1f})')
        plt.axvline(x0_opt, color='b', linestyle='--', label=f'Offset={x0_opt:.2f}MPa')
        
        plt.title(f"Slack Engagement Analysis\nSmoothness (alpha) = {alpha_opt:.2f}")
        plt.xlabel("Pressure (MPa)")
        plt.ylabel("Angle (deg)")
        plt.legend()
        plt.grid(True)
        plt.savefig("slack_analysis_result.png")
        print("グラフを 'slack_analysis_result.png' に保存しました。")
        
    except Exception as e:
        print(f"Fitting Failed: {e}")

if __name__ == "__main__":
    # デフォルトでAntagonisticのデータを解析
    default_file = "data_exp3_slack_pamdf_1768883236.csv" 
    analyze_slack(default_file)