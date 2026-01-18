import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ログファイル名 (スペースに注意して正確に指定してください)
LOG_FILE = "verify_ModelB_Ideal_sine_120bpm.csv"
# LOG_FILE = "simulation_log.csv" # もしファイル名が違う場合はこちら

def plot_hysteresis_v2():
    if not os.path.exists(LOG_FILE):
        print(f"Error: '{LOG_FILE}' not found. Check the filename.")
        return

    # CSVの読み込み
    df = pd.read_csv(LOG_FILE)
    
    # 最初の1秒(不安定な区間)をカット
    df = df[df['time_s'] > 1.0]

    # --- 重要: Action (-1.0 ~ 1.0) を 圧力 (0 ~ 0.6 MPa) に復元する ---
    # action_1 が Flexion (屈曲筋) に対応します
    P_MAX = 0.6
    input_pressure_F = (df['action_1'] + 1.0) / 2.0 * P_MAX

    plt.figure(figsize=(12, 5))

    # --- プロット1: 時系列での比較 ---
    plt.subplot(1, 2, 1)
    plt.plot(df['time_s'], input_pressure_F, 'k--', label='Agent Input (Pure Sine)', linewidth=1.5)
    plt.plot(df['time_s'], df['P_cmd_F'], 'r-', label='Model B Output (P_cmd)', alpha=0.6)
    plt.plot(df['time_s'], df['P_out_F'], 'b-', label='Realized Pressure (P_out)', alpha=0.6)
    
    plt.xlabel('Time [s]')
    plt.ylabel('Pressure [MPa]')
    plt.title('Time Series: Input vs Hysteresis Cmd')
    plt.legend()
    plt.grid(True)

    # --- プロット2: ヒステリシスループ (Input vs P_out) ---
    plt.subplot(1, 2, 2)
    
    # X軸を「加工前のActionから計算した圧力」にするのが正解！
    plt.plot(input_pressure_F, df['P_out_F'], 'g-', alpha=0.6, label='Hysteresis Loop')
    
    # 理想線 (y=x)
    min_val = 0.0
    max_val = 0.6
    plt.plot([min_val, max_val], [min_val, max_val], 'k:', label='Ideal (No Hysteresis)')
    
    plt.xlabel('Agent Input Pressure (Pure Sine)')
    plt.ylabel('Realized Pressure (P_out)')
    plt.title('True Hysteresis Loop Check')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = 'hysteresis_check_v2.png'
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_hysteresis_v2()