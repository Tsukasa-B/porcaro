# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/actions/actuator_net.py
from __future__ import annotations
import torch
import torch.nn as nn
import math
from typing import TYPE_CHECKING
from .layers import PressureNet, ForceNet

if TYPE_CHECKING:
    from ..cfg.actuator_net_cfg import CascadedActuatorNetCfg

class CascadedActuatorNet(nn.Module):
    """
    Model C: Physics-Informed Cascaded ActuatorNet
    3本の人工筋肉 (DF, F, G) を並列処理するPhysics-Informed NN
    """
    def __init__(self, cfg: CascadedActuatorNetCfg, device: str):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
        # --- 1. Neural Networks (Shared Weights) ---
        # 3本すべての筋肉で重みを共有する
        self.pressure_net = PressureNet(
            cfg.pressure_net.input_dim, 
            cfg.pressure_net.output_dim, 
            cfg.pressure_net.hidden_units,
            cfg.pressure_net.activation
        ).to(device)
        
        self.force_net = ForceNet(
            cfg.force_net.input_dim,
            cfg.force_net.output_dim,
            cfg.force_net.hidden_units,
            cfg.force_net.activation
        ).to(device)

        # --- 2. Physics Parameters (Buffer) ---
        # オフセット: (1, 3) にしてブロードキャスト可能にする
        self.register_buffer('slack_offsets', torch.tensor(cfg.slack_offsets, device=device).view(1, 3))
        
        # 幾何学定数
        self.r = cfg.r
        self.L0 = cfg.L0
        # 度 -> ラジアン変換
        refs_rad = [math.radians(x) for x in cfg.theta_ref_deg]
        self.register_buffer('theta_ref', torch.tensor(refs_rad, device=device).view(1, 3))

    def forward(self, 
                p_cmd: torch.Tensor, 
                p_prev: torch.Tensor, 
                theta: torch.Tensor, 
                theta_dot: torch.Tensor):
        """
        Args:
            p_cmd: 指令圧力 [Batch, 3] (DF, F, G)
            p_prev: 前回の圧力 [Batch, 3]
            theta: 関節角度 [Batch, 2] (Wrist, Grip)
            theta_dot: 角速度 [Batch, 2]
        Returns:
            force: 発生張力 [Batch, 3]
            p_est: 推定内圧 [Batch, 3]
        """
        batch_size = p_cmd.shape[0]

        # --- Stage 1: Geometry Calculation (White Box) ---
        # 関節空間(2) -> 筋肉空間(3) への変換
        # epsilon_geo: [Batch, 3]
        eps_geo, eps_vel = self._compute_geometry(theta, theta_dot)
        
        # 有効収縮率 (Slack補正)
        eps_eff = eps_geo - self.slack_offsets

        # --- Stage 2: Pressure Dynamics (NN1) ---
        # 入力をフラットに並べる: [Batch, 3] -> [Batch*3, 1]
        # NN入力: [P_cmd, P_prev, Eps_geo] (すべてBatch*3行)
        p_in = torch.cat([
            p_cmd.view(-1, 1), 
            p_prev.view(-1, 1), 
            eps_geo.view(-1, 1)
        ], dim=1) # shape: [Batch*3, 3]
        
        p_est_flat = self.pressure_net(p_in) # -> [Batch*3, 1]
        p_est = p_est_flat.view(batch_size, 3) # -> [Batch, 3]

        # --- Stage 3: Force Generation (NN2) ---
        # NN入力: [P_est, Eps_eff, Eps_vel]
        f_in = torch.cat([
            p_est_flat,            # 推定圧力
            eps_eff.view(-1, 1),   # 有効収縮率
            eps_vel.view(-1, 1)    # 収縮速度
        ], dim=1) # shape: [Batch*3, 3]

        f_raw_flat = self.force_net(f_in) # -> [Batch*3, 1]
        f_raw = f_raw_flat.view(batch_size, 3) # -> [Batch, 3]
        
        # --- Stage 4: Physics Guard ---
        # たるみ領域(eps_eff < 0)は力ゼロ
        mask = (eps_eff > 0).float()
        force = f_raw * mask
        
        return force, p_est

    def _compute_geometry(self, theta, theta_dot):
        """関節角度(2軸)から筋肉の幾何学的状態(3軸)を計算"""
        # theta: [Batch, 2] -> (Wrist, Grip)
        # theta_muscle: [Batch, 3] -> (DF, F, G)
        
        # マッピング定義:
        # DF (0): Wrist(+)方向で縮む (伸展だが定義上は縮みとして扱うか要確認。通常は屈曲Fが正)
        # ここでは torque.py のロジックに準拠: epsilon = (r/L0) * (theta - theta_ref) * sign
        # DF: Wristを負に動かす力 -> Wristが増えると伸びる -> sign = -1 (または定義による)
        # F : Wristを正に動かす力 -> Wristが増えると縮む -> sign = +1
        # G : Gripを正に動かす力  -> Gripが増えると縮む  -> sign = +1
        
        # Tensor準備
        theta_wrist = theta[:, 0:1]
        theta_grip  = theta[:, 1:2]
        
        # 各筋肉に対応する角度
        # [DF(Wrist), F(Wrist), G(Grip)]
        theta_mapped = torch.cat([theta_wrist, theta_wrist, theta_grip], dim=1)
        
        # 符号 (DFは拮抗側なので逆位相、あるいは参照角との差分絶対値を取る)
        # 簡易実装: epsilon = (r / L0) * (theta - theta_ref) * sign
        # DFは背屈側なので、角度がマイナスに行くほど縮む = (theta_ref - theta)
        # Fは屈曲側なので、角度がプラスに行くほど縮む = (theta - theta_ref)
        
        # 符号ベクトル: [DF=-1, F=1, G=1]
        signs = torch.tensor([-1.0, 1.0, 1.0], device=self.device).view(1, 3)
        
        # 幾何学的収縮率計算
        delta_theta = (theta_mapped - self.theta_ref) * signs
        eps_geo = (self.r / self.L0) * delta_theta
        
        # 速度計算
        theta_dot_wrist = theta_dot[:, 0:1]
        theta_dot_grip  = theta_dot[:, 1:2]
        dot_mapped = torch.cat([theta_dot_wrist, theta_dot_wrist, theta_dot_grip], dim=1)
        eps_vel = (self.r / self.L0) * dot_mapped * signs
        
        return eps_geo, eps_vel