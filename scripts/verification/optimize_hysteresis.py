# scripts/verification/optimize_hysteresis.py
import torch
import torch.nn as nn
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass

# ==========================================
# 1. Config & Model Definitions (Standalone)
# ==========================================
@dataclass
class PamDelayModelCfg:
    tau_pressure_axis: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    tau_values: tuple = (0.043, 0.045, 0.060, 0.066, 0.094, 0.131)
    deadtime_pressure_axis: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    deadtime_values: tuple = (0.038, 0.035, 0.032, 0.030, 0.023, 0.023)
    max_delay_time: float = 0.1
    delay_time: float = 0.04  
    time_constant: float = 0.15

@dataclass
class PamHysteresisModelCfg:
    hysteresis_width: float = 0.1
    curve_shape_param: float = 2.0

class PamDelayModel(nn.Module):
    def __init__(self, cfg: PamDelayModelCfg, dt: float, device: str):
        super().__init__()
        self.cfg = cfg
        self.dt = dt
        self.device = device
        self.register_buffer('tau_p_axis', torch.tensor(self.cfg.tau_pressure_axis, dtype=torch.float32))
        self.register_buffer('tau_vals', torch.tensor(self.cfg.tau_values, dtype=torch.float32))
        self.register_buffer('dead_p_axis', torch.tensor(self.cfg.deadtime_pressure_axis, dtype=torch.float32))
        self.register_buffer('dead_vals', torch.tensor(self.cfg.deadtime_values, dtype=torch.float32))
        self.buffer_len = math.ceil(self.cfg.max_delay_time / self.dt) + 2
        self.pressure_buffer = None
        self.write_ptr = 0
        self.current_pressure = None

    def _init_buffers(self, num_envs, num_channels):
        self.pressure_buffer = torch.zeros((num_envs, num_channels, self.buffer_len), device=self.device)
        self.current_pressure = torch.zeros((num_envs, num_channels), device=self.device)
        self.write_ptr = 0

    def _interp_lut(self, x, xp, fp):
        x_clamped = x.clamp(min=xp[0], max=xp[-1])
        idx = torch.bucketize(x_clamped, xp).clamp(1, len(xp)-1)
        x0, x1 = xp[idx-1], xp[idx]
        f0, f1 = fp[idx-1], fp[idx]
        t = (x_clamped - x0) / (x1 - x0 + 1e-12)
        return f0 + t * (f1 - f0)

    def forward(self, target_pressure):
        if self.pressure_buffer is None: self._init_buffers(target_pressure.shape[0], target_pressure.shape[1])
        L = self._interp_lut(target_pressure, self.dead_p_axis, self.dead_vals)
        tau = self._interp_lut(target_pressure, self.tau_p_axis, self.tau_vals)
        self.pressure_buffer[:, :, self.write_ptr] = target_pressure
        D = (L / self.dt).clamp(min=0.0, max=self.buffer_len - 2.0)
        read_idx_float = (self.write_ptr - D) % self.buffer_len
        idx0 = torch.floor(read_idx_float).long()
        idx1 = (idx0 + 1) % self.buffer_len
        alpha_idx = read_idx_float - idx0
        val0 = self.pressure_buffer.gather(2, idx0.unsqueeze(2)).squeeze(2)
        val1 = self.pressure_buffer.gather(2, idx1.unsqueeze(2)).squeeze(2)
        delayed_input = (1.0 - alpha_idx) * val0 + alpha_idx * val1
        alpha_filter = self.dt / (tau + self.dt)
        self.current_pressure = (1.0 - alpha_filter) * self.current_pressure + alpha_filter * delayed_input
        self.write_ptr = (self.write_ptr + 1) % self.buffer_len
        return self.current_pressure

class PamHysteresisModel(nn.Module):
    def __init__(self, cfg: PamHysteresisModelCfg, device: str):
        super().__init__()
        self.cfg = cfg
        self.last_output = None
    
    def reset_idx(self, env_ids):
        if self.last_output is not None: self.last_output[env_ids] = 0.0

    def forward(self, x):
        if self.last_output is None or self.last_output.shape != x.shape: self.last_output = x.clone()
        r = self.cfg.hysteresis_width / 2.0
        output = torch.max(x - r, torch.min(x + r, self.last_output))
        self.last_output = output.clone()
        return output

# ==========================================
# 2. Optimization Logic
# ==========================================
CSV_PATH = "external_data/jetson_project/01-11-hysteresis.csv"
DT = 0.01

def run_sweep():
    if not os.path.exists(CSV_PATH):
        print("CSV not found.")
        return

    df = pd.read_csv(CSV_PATH)
    cmd_data = torch.tensor(df[['cmd_DF', 'cmd_F', 'cmd_G']].values, dtype=torch.float32)
    meas_data = torch.tensor(df[['meas_pres_DF', 'meas_pres_F', 'meas_pres_G']].values, dtype=torch.float32)
    
    # 探索範囲: 0.00 (Model A相当) 〜 0.15
    widths = np.linspace(0.00, 0.15, 31) 
    rmses = []
    
    best_rmse = float('inf')
    best_width = 0.0

    print(f"Starting sweep on {len(widths)} points...")

    for w in widths:
        # モデル初期化
        cfg_hys = PamHysteresisModelCfg(hysteresis_width=w)
        cfg_delay = PamDelayModelCfg()
        
        model_hys = PamHysteresisModel(cfg_hys, "cpu")
        model_delay = PamDelayModel(cfg_delay, DT, "cpu")
        model_delay._init_buffers(1, 3)

        # シミュレーションループ
        preds = []
        for t in range(len(cmd_data)):
            u = cmd_data[t].unsqueeze(0)
            val = model_hys(u)
            out = model_delay(val)
            preds.append(out.squeeze(0).clone())
        
        pred_tensor = torch.stack(preds)
        
        # DF(背屈)のみで評価
        rmse = torch.sqrt(torch.mean((pred_tensor[:, 0] - meas_data[:, 0])**2)).item()
        rmses.append(rmse)
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_width = w

    # 結果表示
    print(f"\n=== Optimization Result ===")
    print(f"Best Hysteresis Width: {best_width:.4f}")
    print(f"Best RMSE: {best_rmse:.5f}")
    
    # Model A (Width=0) との比較
    rmse_model_a = rmses[0]
    improvement = (rmse_model_a - best_rmse) / rmse_model_a * 100
    print(f"Improvement over Model A: {improvement:.2f}%")

    # プロット
    plt.figure(figsize=(8, 5))
    plt.plot(widths, rmses, 'o-', label='RMSE vs Width')
    plt.axvline(best_width, color='r', linestyle='--', label=f'Best: {best_width:.3f}')
    plt.axhline(rmse_model_a, color='g', linestyle=':', label='Model A (Baseline)')
    plt.xlabel('Hysteresis Width')
    plt.ylabel('RMSE')
    plt.title('Hysteresis Parameter Sweep')
    plt.legend()
    plt.grid()
    plt.savefig('hysteresis_optimization.png')
    print("Saved plot to hysteresis_optimization.png")

if __name__ == "__main__":
    run_sweep()