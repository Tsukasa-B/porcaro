import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def analyze_shape():
    # ---------------------------------------------------------
    # 1. データの読み込み (パス自動解決)
    # ---------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # プロジェクトルートの推定 (porcaro_rl ディレクトリなど)
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # データの検索パス候補
    possible_paths = [
        # コンテナ内やローカル環境の構成に合わせて調整
        os.path.join(project_root, "external_data", "jetson_project", "01-11-hysteresis.csv"),
        os.path.join(project_root, "..", "external_data", "jetson_project", "01-11-hysteresis.csv"),
        # ユーザーのアップロードパス (tsukasa-b/jetson_project/...)
        os.path.join(project_root, "tsukasa-b", "jetson_project", "jetson_project-8e17686396d7192cd0b872be55a0c1aa08554b85", "01-11-hysteresis.csv"),
    ]

    csv_path = None
    for p in possible_paths:
        if os.path.exists(p):
            csv_path = p
            break
            
    if csv_path is None:
        print("[Error] CSV file not found. Please check the path in the script.")
        # 手動でパスを指定したい場合はここを書き換えてください
        # csv_path = "/path/to/your/01-11-hysteresis.csv"
        return

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # ---------------------------------------------------------
    # 2. データ前処理
    # ---------------------------------------------------------
    # 差圧 (Input) と 角度 (Output)
    if 'meas_pres_DF' in df.columns and 'meas_pres_F' in df.columns:
        df['x'] = df['meas_pres_DF'] - df['meas_pres_F']
    else:
        print("[Error] Pressure columns (meas_pres_DF, meas_pres_F) not found.")
        return
        
    df['y'] = df['meas_ang_wrist']
    
    # データの変化方向 (Loading/Unloading) を判定
    # ノイズ除去のため移動平均をとってから微分
    df['x_smooth'] = df['x'].rolling(window=10, center=True).mean()
    df['dx'] = df['x_smooth'].diff()
    
    # 明らかに動いている点だけ抽出 (dxが0付近の静止点を除外)
    threshold = 0.0001
    df_moving = df[df['dx'].abs() > threshold].copy()
    
    # Loading (増圧) と Unloading (減圧) に分割
    loading = df_moving[df_moving['dx'] > 0]
    unloading = df_moving[df_moving['dx'] < 0]

    # ---------------------------------------------------------
    # 3. ループ幅の解析 (Width Analysis)
    # ---------------------------------------------------------
    # 差圧の範囲を3分割して、それぞれの「幅」を計算する
    min_p = df['x'].min()
    max_p = df['x'].max()
    
    # 解析ポイント (低圧域、中圧域、高圧域)
    points = [
        min_p + (max_p - min_p) * 0.25, # Low
        min_p + (max_p - min_p) * 0.50, # Mid
        min_p + (max_p - min_p) * 0.75  # High
    ]
    
    widths = []
    print("\n--- Hysteresis Loop Width Analysis ---")
    print(f"Pressure Range: {min_p:.3f} ~ {max_p:.3f} MPa")
    
    for pt in points:
        # その圧力付近のデータを探す (±0.02MPa)
        window = 0.02
        l_data = loading[(loading['x'] > pt - window) & (loading['x'] < pt + window)]['y']
        u_data = unloading[(unloading['x'] > pt - window) & (unloading['x'] < pt + window)]['y']
        
        if len(l_data) > 0 and len(u_data) > 0:
            avg_load_y = l_data.mean()
            avg_unload_y = u_data.mean()
            
            # 幅 = 角度差 / 感度(概算) で圧力換算も可能だが、ここでは単純に角度差を見る
            width_deg = abs(avg_load_y - avg_unload_y)
            widths.append(width_deg)
            print(f"At Pressure {pt:.3f} MPa: Width = {width_deg:.3f} deg")
        else:
            widths.append(None)
            print(f"At Pressure {pt:.3f} MPa: (Insufficient Data)")

    # ---------------------------------------------------------
    # 4. 判定と結論
    # ---------------------------------------------------------
    print("\n--- Conclusion ---")
    if None in widths:
        print("Cannot determine shape due to insufficient data coverage.")
    else:
        # 幅のばらつきを評価 (標準偏差 / 平均)
        w_avg = np.mean(widths)
        w_std = np.std(widths)
        cv = w_std / w_avg # 変動係数
        
        print(f"Average Width: {w_avg:.3f} deg")
        print(f"Variation (CV): {cv:.3f}")
        
        # 判定ロジック
        if cv < 0.3:
            print(">> JUDGEMENT: Pattern 1 (Constant Width)")
            print("   The loop width is relatively constant across the pressure range.")
            print("   -> Use 'Play Operator' (Friction Model).")
            print("   -> This matches your intuition: 'Friction-based hysteresis'.")
        else:
            if widths[2] > widths[0] * 1.5:
                print(">> JUDGEMENT: Pattern 2 (Fan Shape)")
                print("   The loop gets significantly wider at high pressure.")
                print("   -> Use 'Multiplicative Model' (Ratio Model).")
            else:
                print(">> JUDGEMENT: Complex / Irregular")
                print("   The shape is neither clearly constant nor simply fan-shaped.")
                print("   -> Play Operator is still safer as a baseline.")

if __name__ == "__main__":
    analyze_shape()