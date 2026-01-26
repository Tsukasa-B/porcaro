# source/porcaro_rl/porcaro_rl/utils/actuator_dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import glob
import os
from scipy.signal import savgol_filter

class ActuatorDataset(Dataset):
    def __init__(self, data_dir, seq_len=5, dt=0.01, robot_params=None):
        self.seq_len = seq_len
        self.dt = dt
        
        # 逆動力学用パラメータ (概算)
        self.params = robot_params if robot_params else {
            'I_wrist': 0.005, 'c_wrist': 0.01, 'k_wrist': 0.05,
            'I_hand': 0.001,  'c_hand': 0.005, 'k_hand': 0.01
        }
        
        self.samples = []
        self._load_data(data_dir)

    def _load_data(self, data_dir):
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        # 新しいCSV形式に対応したカラムリスト
        # DF, F, G の順序に注意 (ActuatorNetの内部順序と合わせる: usually [DF, F, G])
        required = ['cmd_DF', 'cmd_F', 'cmd_G', 
                    'meas_pres_DF', 'meas_pres_F', 'meas_pres_G',
                    'meas_ang_wrist', 'meas_ang_hand']

        for f in csv_files:
            try:
                df = pd.read_csv(f)
                if not all(col in df.columns for col in required):
                    continue
                
                # --- 1. 角度データ (2-DOF) ---
                # [Wrist, Hand] の順序で結合
                q_wrist = np.deg2rad(df['meas_ang_wrist'].values)
                q_hand  = np.deg2rad(df['meas_ang_hand'].values)
                theta = np.stack([q_wrist, q_hand], axis=1) # [T, 2]

                # 平滑化と微分
                theta_smooth = np.zeros_like(theta)
                theta_dot = np.zeros_like(theta)
                theta_ddot = np.zeros_like(theta)

                for i in range(2): # 各軸でフィルタ
                    theta_smooth[:, i] = savgol_filter(theta[:, i], 11, 3)
                    theta_dot[:, i]    = savgol_filter(theta[:, i], 11, 3, deriv=1, delta=self.dt)
                    theta_ddot[:, i]   = savgol_filter(theta[:, i], 11, 3, deriv=2, delta=self.dt)

                # --- 2. 圧力データ (3-Muscles) ---
                # ActuatorNetの想定順序 [DF, F, G] に合わせる
                # ※ actuator_net.pyの_compute_geometry等は [DF, F, G] の順で処理していると仮定
                p_cmd = df[['cmd_DF', 'cmd_F', 'cmd_G']].values # [T, 3]
                p_meas = df[['meas_pres_DF', 'meas_pres_F', 'meas_pres_G']].values # [T, 3]

                # --- 3. 正解トルク (Ground Truth Torque) ---
                # Wrist Torque (I * ddq + c * dq + fric)
                tau_wrist = (self.params['I_wrist'] * theta_ddot[:, 0] + 
                             self.params['c_wrist'] * theta_dot[:, 0] + 
                             self.params['k_wrist'] * np.sign(theta_dot[:, 0]))
                
                # Hand Torque
                tau_hand = (self.params['I_hand'] * theta_ddot[:, 1] + 
                            self.params['c_hand'] * theta_dot[:, 1] + 
                            self.params['k_hand'] * np.sign(theta_dot[:, 1]))
                
                tau_gt = np.stack([tau_wrist, tau_hand], axis=1) # [T, 2]

                # シーケンス長チェック
                if len(df) <= self.seq_len: continue

                self.samples.append({
                    'theta': torch.FloatTensor(theta_smooth), # [T, 2]
                    'theta_dot': torch.FloatTensor(theta_dot), # [T, 2]
                    'p_cmd': torch.FloatTensor(p_cmd),        # [T, 3]
                    'p_meas': torch.FloatTensor(p_meas),      # [T, 3]
                    'tau_gt': torch.FloatTensor(tau_gt)       # [T, 2]
                })

            except Exception as e:
                print(f"[Dataset] Skip {f}: {e}")

    def __len__(self):
        return sum(len(s['theta']) - self.seq_len for s in self.samples)

    def __getitem__(self, idx):
        # 簡易ランダムサンプリング (実運用ではEpochごとのShuffle推奨)
        file_idx = np.random.randint(len(self.samples))
        sample = self.samples[file_idx]
        t_idx = np.random.randint(self.seq_len, len(sample['theta']))

        return {
            'theta': sample['theta'][t_idx],         # [2]
            'theta_dot': sample['theta_dot'][t_idx], # [2]
            'p_cmd': sample['p_cmd'][t_idx],         # [3]
            'p_prev': sample['p_meas'][t_idx - 1],   # [3] (1 step old)
            'p_meas_gt': sample['p_meas'][t_idx],    # [3] (Target NN1)
            'tau_gt': sample['tau_gt'][t_idx]        # [2] (Target NN2)
        }