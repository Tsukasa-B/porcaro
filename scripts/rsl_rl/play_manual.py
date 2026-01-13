# scripts/rsl_rl/play_manual.py
"""
Script to play a checkpoint with MANUAL rhythm injection.
Updated: 
1. Fixes ValueError: not enough values to unpack (expected 5, got 4) -> Changed to 4-tuple unpacking.
2. Fixes AttributeError: 'int' object has no attribute 'shape'
3. Adds normalization to match training environment.
4. Removed 'states' keyword argument for rsl_rl compatibility.
"""

import argparse
import sys
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import traceback 

from isaaclab.app import AppLauncher

# local imports
import cli_args

# --- Argument Parser Setup ---
parser = argparse.ArgumentParser(description="Play RL agent with Manual Rhythm Input.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
parser.add_argument("--video_length", type=int, default=400, help="Length of video.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default="Isaac-Porcaro-Direct-v0", help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config.")
parser.add_argument("--seed", type=int, default=None, help="Seed.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use pretrained ckpt.")
parser.add_argument("--real_time", action="store_true", default=False, help="Run simulation in real-time.")

# Manual Rhythm Args
parser.add_argument("--bpm", type=float, default=120.0, help="Target BPM for manual pattern.")
parser.add_argument("--pattern", type=str, default="1,1,1,1", help="Rhythm pattern (1=Hit, 0=Rest).")
parser.add_argument("--grace_period", type=float, default=2.0, help="Silence duration at start [s].")

# RSL-RL args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports after Sim Launch ---
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import porcaro_rl.tasks

# --- Helper Class for Manual Rhythm ---
class ManualRhythmSequencer:
    def __init__(self, pattern: list[int], bpm: float, dt: float, device: str, 
                 target_force: float = 20.0, grace_period: float = 2.0):
        self.device = device
        self.dt = dt
        self.target_force = target_force
        
        self.sp_per_beat = int(60.0 / bpm / dt)
        
        # 1. Grace Period (開始猶予) のステップ数計算
        grace_steps = int(grace_period / dt)
        
        # 2. パターン生成
        # 最低でも20秒分は確保
        min_length_steps = int(20.0 / dt) 
        base_steps = len(pattern) * self.sp_per_beat
        repeat_count = int(np.ceil(min_length_steps / base_steps)) + 1
        
        full_pattern = pattern * repeat_count
        total_steps = grace_steps + len(full_pattern) * self.sp_per_beat + 1000
        
        # 3. スパイク生成
        spikes = torch.zeros((1, 1, total_steps), device=device)
        
        for i, beat_type in enumerate(full_pattern):
            if beat_type == 1:
                idx = grace_steps + (i * self.sp_per_beat)
                if idx < total_steps:
                    spikes[0, 0, idx] = 1.0
                
        # 4. 畳み込み (波形生成)
        width_sec = 0.05
        sigma = width_sec / 2.0
        kernel_radius = int(width_sec / dt)
        t_vals = torch.arange(-kernel_radius, kernel_radius + 1, device=device, dtype=torch.float32) * dt
        
        kernel = target_force * torch.exp(-0.5 * (t_vals / sigma) ** 2)
        kernel = kernel.view(1, 1, -1)
        
        with torch.no_grad():
            traj = F.conv1d(spikes, kernel, padding=kernel_radius)
        
        self.trajectory = traj.squeeze()
        self.max_idx = self.trajectory.shape[0]

    def get_lookahead(self, current_step_idx: int, horizon: int, num_envs: int):
        offsets = torch.arange(horizon, device=self.device)
        indices = current_step_idx + offsets
        valid_mask = indices < self.max_idx
        safe_indices = indices.clamp(max=self.max_idx - 1)
        vals = self.trajectory[safe_indices] * valid_mask.float()
        return vals.unsqueeze(0).repeat(num_envs, 1)

# --- Main Logic ---
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    
    log_dir = os.path.dirname(resume_path)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else "cuda:0"
    
    # プレイ時は強制的にログを有効化
    if hasattr(env_cfg, "logging"):
        env_cfg.logging.enabled = True 

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_manual"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    print(f"[INFO]: Loading model from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # --- Manual Sequencer Setup ---
    dt_ctrl = env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation
    pattern_list = [int(x) for x in args_cli.pattern.split(",")]
    
    sequencer = ManualRhythmSequencer(
        pattern=pattern_list,
        bpm=args_cli.bpm,
        dt=dt_ctrl,
        device=env.unwrapped.device,
        grace_period=args_cli.grace_period
    )
    
    # 修正: .shape[0] を削除して直接 int として取得
    obs_dim = env.unwrapped.cfg.observation_space
    lookahead_steps = obs_dim - 5
    
    print("="*60)
    print(f"[INFO] Manual Rhythm Injection Ready.")
    print(f"  - Pattern: {pattern_list}")
    print(f"  - BPM: {args_cli.bpm}")
    print(f"  - Grace Period: {args_cli.grace_period}s (Silence at start)")
    print("="*60)

    # --- Simulation Loop ---
    obs, _ = env.reset()
    
    # RNNの状態をリセット
    if hasattr(policy, "reset_memory"):
        print("[INFO] Resetting policy RNN memory...")
        policy.reset_memory()
    elif hasattr(policy, "reset"):
        try:
             policy.reset(torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
        except:
             pass

    # 環境設定に合わせた正規化定数
    BPM_NORMALIZATION = 180.0
    FORCE_NORMALIZATION = 20.0

    # 内部バッファの同期（ログ用）
    try:
        if hasattr(env.unwrapped, "rhythm_generator"):
            print("[INFO] Overwriting internal RhythmGenerator for correct logging...")
            rg = env.unwrapped.rhythm_generator
            env_len = rg.target_trajectories.shape[1]
            manual_len = sequencer.trajectory.shape[0]
            copy_len = min(env_len, manual_len)
            rg.target_trajectories[:] = 0.0
            rg.target_trajectories[:, :copy_len] = sequencer.trajectory[:copy_len]
            rg.current_bpms[:] = args_cli.bpm
    except Exception as e:
        print(f"[WARNING] Failed to sync internal rhythm state: {e}")

    sim_step_counter = 0 
    timestep = 0

    print("\n[INFO] Simulation starting...")
    try:
        while simulation_app.is_running():
            start_time = time.time()
            
            with torch.inference_mode():
                if isinstance(obs, dict):
                    obs_tensor = obs["policy"]
                else:
                    obs_tensor = obs

                # --- データの正規化 ---
                obs_tensor[:, 4] = args_cli.bpm / BPM_NORMALIZATION 
                
                manual_traj = sequencer.get_lookahead(
                    current_step_idx=sim_step_counter,
                    horizon=lookahead_steps,
                    num_envs=env.unwrapped.num_envs
                )
                obs_tensor[:, 5:] = manual_traj / FORCE_NORMALIZATION

                # 推論実行 (states引数なし)
                actions = policy(obs_tensor)
                
                # ★修正箇所: 5変数ではなく4変数で受け取る
                obs, _, dones, _ = env.step(actions)

                # エピソード終了時にRNNリセット (donesを使用)
                if dones.any():
                     if hasattr(policy, "reset"):
                         policy.reset(dones)
            
            sim_step_counter += 1
            
            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    print(f"[INFO] Reached video length: {timestep}")
                    break
            
            if args_cli.real_time:
                elapsed = time.time() - start_time
                if dt_ctrl > elapsed:
                    time.sleep(dt_ctrl - elapsed)

    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        print(f"\n[ERROR] Exception occurred during simulation loop:")
        traceback.print_exc()
    finally:
        print("[INFO] Closing environment...")
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()