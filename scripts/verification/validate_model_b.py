import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ==========================================
# 取得した係数 (ユーザーの解析結果)
# ==========================================
# Angle = a*x^3 + b*x^2 + c*x + d
# x = P_DF - P_F
COEFFS_LOAD   = [12.89864007, -53.55390049, -71.00674403, -14.42740187]
COEFFS_UNLOAD = [27.92874623, -39.23796728, -74.33612987, -18.54505562]

def validate_step_response():
    # 1. Stepデータの検索
    files = glob.glob('**/01-11-step.csv', recursive=True)
    if not files:
        files = glob.glob('**/*step.csv', recursive=True)
    
    if not files:
        print("Error: Step response file not found.")
        return

    target_file = files[0]
    print(f"Validating against: {target_file}")
    df = pd.read_csv(target_file)

    # 2. 差圧入力の計算
    # 実機ログから入力を生成
    if 'meas_pres_DF' not in df.columns or 'meas_pres_F' not in df.columns:
        print("Error: Pressure columns not found.")
        return
        
    df['p_diff'] = df['meas_pres_DF'] - df['meas_pres_F']
    
    # 3. モデルによる予測計算
    pred_angles = []
    
    # 状態判定用: 差圧の変化速度
    df['dp'] = df['p_diff'].diff().fillna(0)
    
    for i, row in df.iterrows():
        x = row['p_diff']
        dp = row['dp']
        
        # Loading (増圧) か Unloading (減圧) かで式を切り替え
        # ※ノイズで頻繁に切り替わるのを防ぐため、本来は「状態遷移ロジック」が必要だが
        #   ここでは単純に瞬時の速度で判定する
        if dp >= 0:
            # Loading
            y = np.polyval(COEFFS_LOAD, x)
        else:
            # Unloading
            y = np.polyval(COEFFS_UNLOAD, x)
            
        pred_angles.append(y)

    df['pred_angle'] = pred_angles

    # 4. プロット比較
    plt.figure(figsize=(12, 6))
    
    # 時系列での比較
    plt.subplot(1, 1, 1)
    plt.plot(df.index, df['meas_ang_wrist'], 'k-', alpha=0.6, label='Real Robot (Ground Truth)')
    plt.plot(df.index, df['pred_angle'], 'r--', linewidth=2, label='Model B Prediction')
    
    plt.title("Validation: Real Robot vs Simple Hysteresis Model (Model B)")
    plt.xlabel("Time steps")
    plt.ylabel("Wrist Angle [deg]")
    plt.legend()
    plt.grid(True)
    
    save_path = "validation_model_b_step.png"
    plt.savefig(save_path)
    print(f"Validation plot saved to: {save_path}")
    print("グラフの赤点線(モデル)が黒実線(実機)を追従できていれば成功です。")

if __name__ == "__main__":
    validate_step_response()