import pandas as pd
import matplotlib.pyplot as plt

# ファイル名 (スペースが入らないように注意)
file_a = "verify_ModelA_Ideal_REPLAY_01-11-hysteresis.csv"
file_b = "verify_ModelB_Ideal_REPLAY_01-11-hysteresis.csv"

def check_difference():
    # データの読み込み
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)

    # タイムスタンプが一致しているか確認しつつ、データ長を合わせる
    min_len = min(len(df_a), len(df_b))
    df_a = df_a.iloc[:min_len]
    df_b = df_b.iloc[:min_len]

    # 時間軸
    time = df_a['time_s']

    # 角度の差分を計算 (Model B - Model A)
    diff_wrist = df_b['q_wrist_deg'] - df_a['q_wrist_deg']
    
    # プロット
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(time, df_a['q_wrist_deg'], 'b--', label='Model A (No Hysteresis)', alpha=0.7)
    plt.plot(time, df_b['q_wrist_deg'], 'r-', label='Model B (Hysteresis)', alpha=0.7)
    plt.title("Joint Angle Comparison")
    plt.ylabel("Wrist Angle [deg]")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(time, diff_wrist, 'g-')
    plt.title("Difference (Model B - Model A)")
    plt.ylabel("Delta Angle [deg]")
    plt.xlabel("Time [s]")
    plt.grid()
    
    # 差が0なら警告
    if diff_wrist.abs().max() < 0.001:
        print("警告: 動きが完全に一致しています。P_outが適用されていない可能性があります。")
    else:
        print(f"確認: 最大 {diff_wrist.abs().max():.2f} 度のズレが生じています。反映されています！")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    check_difference()