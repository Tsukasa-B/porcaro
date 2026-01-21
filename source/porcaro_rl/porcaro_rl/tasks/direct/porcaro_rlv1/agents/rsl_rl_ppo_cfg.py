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
    # ★変更: 16 -> 48
    # 50Hzで約1秒分（2拍分）の未来まで見通して、次の動作を計画させます。
    num_steps_per_env = 48
    
    # ★変更: 150 -> 1500
    # 長時間のエピソードで安定したリズムを習得するため、試行回数を増やします。
    max_iterations = 1500
    
    save_interval = 50
    experiment_name = "porcaro_rslrl_recurrent_lstm_double" # 名前を変えておくと管理しやすいです
    
    # 💡 修正点2: PolicyクラスをRecurrentバージョンに変更
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        # RNNを使う場合、観測の正規化をONにすることが推奨されます
        actor_obs_normalization=True, 
        critic_obs_normalization=True, 
        
        # ネットワークサイズを大きめに設定 (RNNの隠れ層サイズと揃えることが多い)
        actor_hidden_dims=[128, 128], 
        critic_hidden_dims=[128, 128], 
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
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )