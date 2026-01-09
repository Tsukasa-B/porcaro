import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    # 引数設定
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="log_double_80bpm_NoDrum.csv", help="CSV file path")
    args = parser.parse_args()

    # CSV読み込み
    try:
        df = pd.read_csv(args.file)
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found.")
        return

    # 時間軸
    t = df['time_s']

    # プロット作成 (3段)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 1. 圧力 (Pressure): 指令 vs 実際
    ax = axes[0]
    ax.plot(t, df['P_cmd_F'], label='Cmd (Flexion)', color='red', linestyle='--', alpha=0.7)
    ax.plot(t, df['P_out_F'], label='Out (Flexion)', color='red')
    ax.plot(t, df['P_cmd_DF'], label='Cmd (Extension)', color='blue', linestyle='--', alpha=0.7)
    ax.plot(t, df['P_out_DF'], label='Out (Extension)', color='blue')
    ax.set_ylabel('Pressure [MPa]')
    ax.set_title('Pressure Dynamics (Command vs Output)')
    ax.legend(loc='upper right')
    ax.grid(True)

    # 2. 角度 (Angle): 手首の動き
    ax = axes[1]
    ax.plot(t, df['q_wrist_deg'], label='Wrist Angle', color='black')
    ax.set_ylabel('Angle [deg]')
    ax.set_title('Wrist Motion')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.legend()
    ax.grid(True)

    # 3. トルク (Torque): 発生トルク
    ax = axes[2]
    ax.plot(t, df['tau_wrist'], label='Wrist Torque', color='green')
    ax.set_ylabel('Torque [Nm]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Generated Torque')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("verification_result.png")
    print("Graph saved to 'verification_result.png'")
    # plt.show() # GUI環境ならコメントアウトを外す

if __name__ == "__main__":
    main()