import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 設定 ---
REWARD_LOG_FILE = 'reward_log.csv'
SIM_LOG_FILE = 'simulation_log.csv'
SAVE_PLOTS = False  # 画像として保存する場合はTrueにしてください

def plot_rewards():
    if not os.path.exists(REWARD_LOG_FILE):
        print(f"File not found: {REWARD_LOG_FILE}")
        return

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
    plt.show()

def plot_simulation():
    if not os.path.exists(SIM_LOG_FILE):
        print(f"File not found: {SIM_LOG_FILE}")
        return

    df = pd.read_csv(SIM_LOG_FILE)
    time = df['time_s']

    # --- 1. Actions (AI指令値) ---
    fig1, ax1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig1.suptitle('Actions (AI Output)')
    
    ax1[0].plot(time, df['act_theta_eq'], color='tab:blue')
    ax1[0].set_ylabel('Theta Eq (Action)')
    ax1[0].grid(True)
    
    ax1[1].plot(time, df['act_K_wrist'], color='tab:orange')
    ax1[1].set_ylabel('K Wrist (Action)')
    ax1[1].grid(True)
    
    ax1[2].plot(time, df['act_K_grip'], color='tab:green')
    ax1[2].set_ylabel('K Grip (Action)')
    ax1[2].set_xlabel('Time [s]')
    ax1[2].grid(True)

    if SAVE_PLOTS: fig1.savefig('sim_actions.png')

    # --- 2. Joint States (角度・角速度) ---
    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig2.suptitle('Joint States')

    ax2[0].plot(time, df['q_wrist'], label='Wrist', color='blue')
    ax2[0].plot(time, df['q_grip'], label='Grip', color='green')
    ax2[0].set_ylabel('Position [rad]')
    ax2[0].legend()
    ax2[0].grid(True)

    ax2[1].plot(time, df['qd_wrist'], label='Wrist Vel', color='blue', linestyle='--')
    ax2[1].plot(time, df['qd_grip'], label='Grip Vel', color='green', linestyle='--')
    ax2[1].set_ylabel('Velocity [rad/s]')
    ax2[1].legend()
    ax2[1].grid(True)
    ax2[1].set_xlabel('Time [s]')

    if SAVE_PLOTS: fig2.savefig('sim_joints.png')

    # --- 3. Pressures (指令値 vs 実測値) ---
    fig3, ax3 = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig3.suptitle('Pneumatic Pressures (Command vs Output)')

    # Wrist DF (Extensor)
    ax3[0].plot(time, df['P_cmd_DF'], label='Cmd DF', color='red', linestyle='--')
    ax3[0].plot(time, df['P_out_DF'], label='Out DF', color='red', alpha=0.7)
    ax3[0].set_ylabel('Pressure [MPa]')
    ax3[0].legend(loc='upper right')
    ax3[0].grid(True)
    ax3[0].set_title('Wrist DF (Extensor)')

    # Wrist F (Flexor)
    ax3[1].plot(time, df['P_cmd_F'], label='Cmd F', color='blue', linestyle='--')
    ax3[1].plot(time, df['P_out_F'], label='Out F', color='blue', alpha=0.7)
    ax3[1].set_ylabel('Pressure [MPa]')
    ax3[1].legend(loc='upper right')
    ax3[1].grid(True)
    ax3[1].set_title('Wrist F (Flexor)')

    # Grip
    ax3[2].plot(time, df['P_cmd_G'], label='Cmd G', color='green', linestyle='--')
    ax3[2].plot(time, df['P_out_G'], label='Out G', color='green', alpha=0.7)
    ax3[2].set_ylabel('Pressure [MPa]')
    ax3[2].set_xlabel('Time [s]')
    ax3[2].legend(loc='upper right')
    ax3[2].grid(True)
    ax3[2].set_title('Grip G')

    if SAVE_PLOTS: fig3.savefig('sim_pressures.png')

    # --- 4. Forces & Torques ---
    fig4, ax4 = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig4.suptitle('Forces and Torques')

    # Stick Contact Force
    ax4[0].plot(time, df['stick_force_z'], label='Force Z', color='purple')
    # ax4[0].plot(time, df['stick_force_x'], label='Force X', alpha=0.3) # 必要ならコメントアウト解除
    # ax4[0].plot(time, df['stick_force_y'], label='Force Y', alpha=0.3)
    ax4[0].plot(time, df['f1_force'], label='F1 Score (Peak)', color='orange', linestyle='--', linewidth=2)
    ax4[0].set_ylabel('Force [N]')
    ax4[0].legend()
    ax4[0].grid(True)
    ax4[0].set_title('Stick Contact Force & F1 Score')

    # Wrist Torque
    ax4[1].plot(time, df['tau_w'], color='blue')
    ax4[1].set_ylabel('Wrist Torque [Nm]')
    ax4[1].grid(True)

    # Grip Torque
    ax4[2].plot(time, df['tau_g'], color='green')
    ax4[2].set_ylabel('Grip Torque [Nm]')
    ax4[2].set_xlabel('Time [s]')
    ax4[2].grid(True)

    if SAVE_PLOTS: fig4.savefig('sim_forces_torques.png')

    plt.show()

if __name__ == "__main__":
    print("Plotting Reward Log...")
    plot_rewards()
    
    print("Plotting Simulation Log...")
    plot_simulation()