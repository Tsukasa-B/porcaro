import pandas as pd
import matplotlib.pyplot as plt

# データの読み込み
df_sim = pd.read_csv('simulation_log.csv')

# エピソードIDを付与
df_sim['episode_id'] = (df_sim['time_s'].diff() < 0).cumsum()

# 最終エピソードのデータを抽出
last_episode_data = df_sim[df_sim['episode_id'] == df_sim['episode_id'].max()]

# グラフ描画
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# 1. 関節角度 (Motion)
axes[0].plot(last_episode_data['time_s'], last_episode_data['q_wrist'], label='Wrist Angle (rad)', color='blue')
axes[0].plot(last_episode_data['time_s'], last_episode_data['q_grip'], label='Grip Angle (rad)', color='orange', linestyle='--')
axes[0].set_title('Joint Angles (Motion)', fontsize=14)
axes[0].set_ylabel('Angle [rad]', fontsize=12)
axes[0].legend(loc='upper right')
axes[0].grid(True, linestyle='--', alpha=0.7)

# 2. 打撃力 (Impact)
axes[1].plot(last_episode_data['time_s'], last_episode_data['stick_force_z'], label='Stick Force Z (Impact)', color='red')
axes[1].set_title('Impact Force', fontsize=14)
axes[1].set_ylabel('Force [N]', fontsize=12)
axes[1].legend(loc='upper right')
axes[1].grid(True, linestyle='--', alpha=0.7)

# 3. 圧力指令 (Intention)
axes[2].plot(last_episode_data['time_s'], last_episode_data['P_cmd_DF'], label='P_cmd DF (Extensor)', color='cyan')
axes[2].plot(last_episode_data['time_s'], last_episode_data['P_cmd_F'], label='P_cmd F (Flexor)', color='magenta')
axes[2].set_title('Pressure Commands (AI Output)', fontsize=14)
axes[2].set_xlabel('Time [s]', fontsize=12)
axes[2].set_ylabel('Pressure [MPa]', fontsize=12)
axes[2].legend(loc='upper right')
axes[2].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()