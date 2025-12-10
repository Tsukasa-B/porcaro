import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# データの読み込み
df = pd.read_csv('reward_log.csv')

# --- 1. エピソードの区切りを検出 ---
# time_s が若返った（リセットされた）場所を見つける
# diff() がマイナスになる場所がエピソードの変わり目
df['episode_id'] = (df['time_s'].diff() < 0).cumsum()

# --- 2. エピソードごとの集計 ---
# エピソードごとの「合計報酬」と「平均報酬」を計算
episode_stats = df.groupby('episode_id')['step_reward'].agg(['sum', 'mean', 'count'])

# --- 3. グラフ描画 ---
plt.figure(figsize=(14, 10))

# 上段: 学習曲線 (Episode Total Reward)
plt.subplot(2, 1, 1)
plt.plot(episode_stats.index, episode_stats['sum'], label='Episode Total Reward', color='blue', linewidth=1.5)
plt.title('Learning Curve: Total Reward per Episode', fontsize=14)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Total Reward', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 下段: 報酬の時系列詳細 (最後の1エピソードを拡大)
# 学習の最後で「どんなリズムで叩いていたか」を確認
last_episode_id = df['episode_id'].max()
last_episode_data = df[df['episode_id'] == last_episode_id]

plt.subplot(2, 1, 2)
plt.plot(last_episode_data['time_s'], last_episode_data['step_reward'], color='green')
plt.title(f'Rhythm Pattern in Last Episode (Ep #{last_episode_id})', fontsize=14)
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Step Reward (Hit Quality)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# --- 4. 数値レポート ---
print("-" * 40)
print(f"📊 学習評価レポート")
print("-" * 40)
print(f"総エピソード数: {len(episode_stats)}")
print(f"最初のエピソードのスコア: {episode_stats['sum'].iloc[0]:.2f}")
print(f"最後のエピソードのスコア: {episode_stats['sum'].iloc[-1]:.2f}")

# スコアが向上しているか判定
improvement = episode_stats['sum'].iloc[-1] - episode_stats['sum'].iloc[0]
if improvement > 0:
    print(f"✅ 判定: 学習成功！ スコアが +{improvement:.2f} 向上しました。")
else:
    print(f"⚠️ 判定: スコアが伸び悩んでいます。報酬設定や学習回数の見直しが必要かもしれません。")
print("-" * 40)