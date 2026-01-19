# scripts/verification/compare_models_standalone.py
import torch
import torch.nn as nn
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass

# ==========================================
# 1. Config Definitions (Mocking Isaac Lab Configs)
# ==========================================
@dataclass
class PamDelayModelCfg:
    # 圧力軸 [MPa]
    tau_pressure_axis: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    # 時定数値 [s]
    tau_values: tuple = (0.043, 0.045, 0.060, 0.066, 0.094, 0.131)
    # むだ時間軸 [MPa]
    deadtime_pressure_axis: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    # むだ時間値 [s]
    deadtime_values: tuple = (0.038, 0.035, 0.032, 0.030, 0.023, 0.023)
    max_delay_time: float = 0.1
    # Legacy params
    delay_time: float = 0.04  
    time_constant: float = 0.15

@dataclass
class PamHysteresisModelCfg:
    hysteresis_width: float = 0.1
    curve_shape_param: float = 2.0

# ==========================================
# 2. Model Implementations (Copied from pam_dynamics.py)
#    Removed isaaclab imports for standalone execution
# ==========================================
class PamDelayModel(nn.Module):
    def __init__(self, cfg: PamDelayModelCfg, dt: float, device: str):
        super().__init__()
        self.cfg = cfg
        self.dt = dt
        self.device = device
        
        # LUT registration
        self.register_buffer('tau_p_axis', torch.tensor(
            self.cfg.tau_pressure_axis, dtype=torch.float32, device=self.device))
        self.register_buffer('tau_vals', torch.tensor(
            self.cfg.tau_values, dtype=torch.float32, device=self.device))

        self.register_buffer('dead_p_axis', torch.tensor(
            self.cfg.deadtime_pressure_axis, dtype=torch.float32, device=self.device))
        self.register_buffer('dead_vals', torch.tensor(
            self.cfg.deadtime_values, dtype=torch.float32, device=self.device))

        # Ring buffer setup
        max_delay = self.cfg.max_delay_time
        self.buffer_len = math.ceil(max_delay / self.dt) + 2
        
        self.pressure_buffer = None
        self.write_ptr = 0
        self.current_pressure = None

    def _init_buffers(self, num_envs: int, num_channels: int):
        self.pressure_buffer = torch.zeros((num_envs, num_channels, self.buffer_len), device=self.device)
        self.current_pressure = torch.zeros((num_envs, num_channels), device=self.device)
        self.write_ptr = 0

    def reset_idx(self, env_ids: torch.Tensor):
        if self.pressure_buffer is not None:
            self.pressure_buffer[env_ids] = 0.0
            self.current_pressure[env_ids] = 0.0

    def _interp_lut(self, x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        x_clamped = x.clamp(min=xp[0], max=xp[-1])
        idx = torch.bucketize(x_clamped, xp).clamp(1, len(xp)-1)
        x0, x1 = xp[idx-1], xp[idx]
        f0, f1 = fp[idx-1], fp[idx]
        t = (x_clamped - x0) / (x1 - x0 + 1e-12)
        return f0 + t * (f1 - f0)

    def forward(self, target_pressure: torch.Tensor) -> torch.Tensor:
        if self.pressure_buffer is None:
            self._init_buffers(target_pressure.shape[0], target_pressure.shape[1])
            
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
        self.device = device
        self.last_output = None

    def reset_idx(self, env_ids: torch.Tensor):
        if self.last_output is not None:
            self.last_output[env_ids] = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.last_output is None or self.last_output.shape != x.shape:
            self.last_output = x.clone()

        r = self.cfg.hysteresis_width / 2.0
        lower_bound = x - r
        upper_bound = x + r
        
        output = torch.max(lower_bound, torch.min(upper_bound, self.last_output))
        self.last_output = output.clone()
        return output

# ==========================================
# 3. Main Verification Logic
# ==========================================
# --- Parameters ---
ESTIMATED_WIDTH = 0.08  # Step 1で求めた値をここに入力
DT = 0.01               # 10ms
CSV_PATH = "external_data/jetson_project/01-11-hysteresis.csv" # パス確認

def run_comparison():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        return

    # Load Data
    df = pd.read_csv(CSV_PATH)
    # 背屈(DF), 屈曲(F), 握り(G)
    cmd_cols = ['cmd_DF', 'cmd_F', 'cmd_G']
    meas_cols = ['meas_pres_DF', 'meas_pres_F', 'meas_pres_G']
    
    # Check if columns exist
    if not all(col in df.columns for col in cmd_cols + meas_cols):
        print("Error: Columns mismatch in CSV.")
        return

    cmd_data = torch.tensor(df[cmd_cols].values, dtype=torch.float32)
    meas_data = torch.tensor(df[meas_cols].values, dtype=torch.float32)
    
    num_samples = cmd_data.shape[0]
    num_channels = 3 

    # --- Setup Models ---
    # Model A: Delay Only
    cfg_a = PamDelayModelCfg()
    model_a = PamDelayModel(cfg_a, DT, "cpu")
    
    # Model B: Hysteresis + Delay
    cfg_b_hys = PamHysteresisModelCfg(hysteresis_width=ESTIMATED_WIDTH)
    cfg_b_delay = PamDelayModelCfg()
    model_b_hys = PamHysteresisModel(cfg_b_hys, "cpu")
    model_b_delay = PamDelayModel(cfg_b_delay, DT, "cpu")

    # --- Run Simulation ---
    out_a_list = []
    out_b_list = []

    # Reset
    # Note: reset_idx logic slightly simplified for batch 1 implicit
    model_a._init_buffers(1, num_channels)
    model_b_delay._init_buffers(1, num_channels)
    
    print("Running simulation...")
    for t in range(num_samples):
        # [1, 3] batch size 1
        u = cmd_data[t].unsqueeze(0) 

        # Model A
        y_a = model_a(u)
        out_a_list.append(y_a.squeeze(0).clone())

        # Model B
        val_hys = model_b_hys(u)
        y_b = model_b_delay(val_hys)
        out_b_list.append(y_b.squeeze(0).clone())

    pred_a = torch.stack(out_a_list)
    pred_b = torch.stack(out_b_list)

    # --- Analysis (Focus on DF channel) ---
    ch_idx = 0 # DF
    rmse_a = torch.sqrt(torch.mean((pred_a[:, ch_idx] - meas_data[:, ch_idx])**2)).item()
    rmse_b = torch.sqrt(torch.mean((pred_b[:, ch_idx] - meas_data[:, ch_idx])**2)).item()

    print(f"=== Results (Channel DF) ===")
    print(f"Model A RMSE: {rmse_a:.5f}")
    print(f"Model B RMSE: {rmse_b:.5f}")
    print(f"Improvement: {(rmse_a - rmse_b)/rmse_a * 100:.2f}%")

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    
    # Time Series
    plt.subplot(1, 2, 1)
    # Zoom in to a dynamic part if possible, or first 500 steps
    zoom = slice(0, min(500, num_samples))
    plt.plot(meas_data[zoom, ch_idx].numpy(), 'k-', lw=2, alpha=0.3, label='Real')
    plt.plot(pred_a[zoom, ch_idx].numpy(), 'b--', label='Model A')
    plt.plot(pred_b[zoom, ch_idx].numpy(), 'r-', label='Model B')
    plt.title("Time Series Response (Zoomed)")
    plt.legend()
    plt.grid()

    # Hysteresis Loop
    plt.subplot(1, 2, 2)
    plt.plot(cmd_data[:, ch_idx].numpy(), meas_data[:, ch_idx].numpy(), 'k', alpha=0.1, label='Real')
    plt.plot(cmd_data[:, ch_idx].numpy(), pred_a[:, ch_idx].numpy(), 'b--', alpha=0.6, label='Model A')
    plt.plot(cmd_data[:, ch_idx].numpy(), pred_b[:, ch_idx].numpy(), 'r-', alpha=0.6, label='Model B')
    plt.title(f"Hysteresis Loop (Width={ESTIMATED_WIDTH})")
    plt.xlabel("Command Pressure")
    plt.ylabel("Measured Pressure")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("comparison_standalone_result.png")
    print("Saved plot to comparison_standalone_result.png")

if __name__ == "__main__":
    run_comparison()