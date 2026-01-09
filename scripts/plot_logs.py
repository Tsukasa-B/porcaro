# scripts/plot_logs.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

# --- 設定 ---
REWARD_LOG_FILE = 'reward_log.csv'
SIM_LOG_FILE = 'simulation_log.csv'
SAVE_PLOTS = False  # 画像として保存する場合はTrue

def plot_rewards():
    if not os.path.exists(REWARD_LOG_FILE):
        print(f"[WARN] File not found: {REWARD_LOG_FILE}")
        return

    try:
        df = pd.read_csv(REWARD_LOG_FILE)
        
        plt.figure(figsize=(10, 5))
        plt.plot(df['time_s'], df['step_reward'], label='Step Reward', color='blue', alpha=0.7)
        plt.xlabel('Time [s]')
        plt.ylabel('Reward')
        plt.title('Reward History')
        plt.grid(True)
        plt.legend()
        
        if SAVE_PLOTS:
            plt.savefig('reward_plot.png')
            print("[INFO] Saved reward_plot.png")
        #plt.show()
    except Exception as e:
        print(f"[ERROR] Failed to plot rewards: {e}")

def plot_simulation():
    if not os.path.exists(SIM_LOG_FILE):
        print(f"[WARN] File not found: {SIM_LOG_FILE}")
        return

    try:
        df = pd.read_csv(SIM_LOG_FILE)
        time = df['time_s']

        # --- 1. Actions (AI指令値) ---
        fig1, ax1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig1.suptitle('Actions (AI Output / Normalized)')
        
        # Action 0 (Theta_eq or P_DF)
        ax1[0].plot(time, df['action_0'], label='Action[0]', color='blue')
        ax1[0].set_ylabel('Value')
        ax1[0].legend(loc='upper right')
        ax1[0].grid(True)
        ax1[0].set_title('Action[0] (Theta_eq / P_DF)')

        # Action 1 (K_wrist or P_F)
        ax1[1].plot(time, df['action_1'], label='Action[1]', color='green')
        ax1[1].set_ylabel('Value')
        ax1[1].legend(loc='upper right')
        ax1[1].grid(True)
        ax1[1].set_title('Action[1] (Stiffness / P_F)')

        # Action 2 (K_grip or P_G)
        ax1[2].plot(time, df['action_2'], label='Action[2]', color='red')
        ax1[2].set_ylabel('Value')
        ax1[2].set_xlabel('Time [s]')
        ax1[2].legend(loc='upper right')
        ax1[2].grid(True)
        ax1[2].set_title('Action[2] (Grip / P_G)')

        if SAVE_PLOTS: fig1.savefig('sim_actions.png')

        # --- 2. Joint Angles (Degrees) ---
        fig2, ax2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig2.suptitle('Joint Angles [deg] (Should be smooth!)')

        ax2[0].plot(time, df['q_wrist_deg'], label='Wrist', color='blue')
        ax2[0].set_ylabel('Angle [deg]')
        ax2[0].legend()
        ax2[0].grid(True)
        
        ax2[1].plot(time, df['q_grip_deg'], label='Grip', color='green')
        ax2[1].set_ylabel('Angle [deg]')
        ax2[1].set_xlabel('Time [s]')
        ax2[1].legend()
        ax2[1].grid(True)

        if SAVE_PLOTS: fig2.savefig('sim_angles.png')

        # --- 3. Pressures (Dynamics Check) ---
        # 指令(Cmd)と応答(Out)を重ねて表示し、遅延を確認する
        fig3, ax3 = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig3.suptitle('Internal Pressures [MPa] (Dashed: Cmd, Solid: Real)')

        # Wrist DF
        ax3[0].plot(time, df['P_cmd_DF'], label='Cmd DF', color='blue', linestyle='--', alpha=0.8)
        ax3[0].plot(time, df['P_out_DF'], label='Out DF', color='blue', alpha=1.0)
        ax3[0].set_ylabel('Pressure [MPa]')
        ax3[0].legend(loc='upper right')
        ax3[0].grid(True)
        ax3[0].set_title('Wrist DF (Flexor)')

        # Wrist F
        ax3[1].plot(time, df['P_cmd_F'], label='Cmd F', color='red', linestyle='--', alpha=0.8)
        ax3[1].plot(time, df['P_out_F'], label='Out F', color='red', alpha=1.0)
        ax3[1].set_ylabel('Pressure [MPa]')
        ax3[1].legend(loc='upper right')
        ax3[1].grid(True)
        ax3[1].set_title('Wrist F (Extensor)')

        # Grip G
        ax3[2].plot(time, df['P_cmd_G'], label='Cmd G', color='green', linestyle='--', alpha=0.8)
        ax3[2].plot(time, df['P_out_G'], label='Out G', color='green', alpha=1.0)
        ax3[2].set_ylabel('Pressure [MPa]')
        ax3[2].set_xlabel('Time [s]')
        ax3[2].legend(loc='upper right')
        ax3[2].grid(True)
        ax3[2].set_title('Grip G')

        if SAVE_PLOTS: fig3.savefig('sim_pressures.png')

        # --- 4. Forces & Torques ---
        fig4, ax4 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig4.suptitle('Forces and Torques')

        # Stick Contact Force (Z-axis only)
        ax4[0].plot(time, df['force_z'], label='Force Z', color='purple')
        ax4[0].plot(time, df['f1_score'], label='F1 Score (Peak)', color='orange', linestyle='--', linewidth=2)
        ax4[0].set_ylabel('Force [N]')
        ax4[0].legend()
        ax4[0].grid(True)
        ax4[0].set_title('Stick Contact Force & F1 Score')

        # Joint Torques
        ax4[1].plot(time, df['tau_wrist'], label='Wrist Torque', color='blue', alpha=0.7)
        ax4[1].plot(time, df['tau_grip'], label='Grip Torque', color='green', alpha=0.7)
        ax4[1].set_ylabel('Torque [Nm]')
        ax4[1].set_xlabel('Time [s]')
        ax4[1].legend()
        ax4[1].grid(True)
        ax4[1].set_title('Joint Torques')

        if SAVE_PLOTS: fig4.savefig('sim_forces.png')

        

    except KeyError as e:
        print(f"[ERROR] Key not found in CSV: {e}")
        print("ヘッダーが古い可能性があります。simulation_log.csv を削除して再生成してください。")
    except Exception as e:
        print(f"[ERROR] Failed to plot simulation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true', help='Save plots as images')
    args = parser.parse_args()
    
    if args.save:
        SAVE_PLOTS = True
        
    print("Plotting results...")
    plot_simulation()
    plot_rewards() # 必要に応じてコメントアウト解除
    plt.show()