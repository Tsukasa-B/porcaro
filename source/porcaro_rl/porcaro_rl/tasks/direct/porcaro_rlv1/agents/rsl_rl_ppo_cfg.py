# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rl/agents/rsl_rl_ppo_cfg.py

from isaaclab.utils import configclass

# 💡 修正点1: RecurrentCfgをインポート（または置き換え）
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg, 
    RslRlPpoAlgorithmCfg, 
    RslRlPpoActorCriticRecurrentCfg # 新しいクラス
) 


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # ★ 改善: 1秒(48) -> 2.4秒(120) 程度まで伸ばす
    # BPM60で2拍以上、BPM120なら1小節分を見渡せるようにする
    num_steps_per_env = 120
    
    # ★変更: 150 -> 1500
    # 長時間のエピソードで安定したリズムを習得するため、試行回数を増やします。
    max_iterations = 1000
    
    save_interval = 50
    experiment_name = "porcaro_rslrl_lstm_modelB_DR" # 名前を変えておくと管理しやすいです
    
    # 💡 修正点2: PolicyクラスをRecurrentバージョンに変更
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=0.5,
        # RNNを使う場合、観測の正規化をONにすることが推奨されます
        actor_obs_normalization=True, 
        critic_obs_normalization=True, 
        
        # ネットワークサイズを大きめに設定 (RNNの隠れ層サイズと揃えることが多い)
        # ★ ネットワーク構成
        # [400, 200, 100] くらいが一般的だが、タスクが複雑ならこのままでもOK
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        
        # 💡 修正点3: RNN関連の引数を設定
        rnn_type="lstm", # or "gru"
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001, # 探索がすぐ収束してしまうようなら 0.01 -> 0.02 に上げる
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )