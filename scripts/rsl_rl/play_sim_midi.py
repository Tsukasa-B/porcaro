# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/scripts/rsl_rl/play_sim_midi.py
# (または保存されているディレクトリの同名ファイル)

"""
Play script with MIDI Injection (Version 4.5 - Auto Exit)
修正点:
1. ループ動作を廃止。
2. 「曲完走」または「転倒/リセット」でスクリプトを自動終了するように変更。
3. 波形生成をスーパーガウシアン（フラットトップ波形）にアップデート。

Usage:
  python scripts/rsl_rl/play_sim_midi.py \
  --task Template-Porcaro-Direct-ModelB \
  --experiment porcaro_rslrl_lstm_modelB_DR \
  --load_run 2026-02-15_20-55-46 \
  --midi songs/test_single8_bpm120.mid \
  --video

  python scripts/rsl_rl/play_sim_midi.py   --task Template-Porcaro-Direct-ModelB   --experiment porcaro_rslrl_lstm_modelB_DR   --load_run 2026-02-25_22-28-09  --midi songs/test_single8_bpm160.mid   --video
"""

import argparse
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

# --- Fix 1: Add script directory to sys.path for local imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from isaaclab.app import AppLauncher
import cli_args

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Play RL agent with MIDI Input (Injection Mode).")

# 1. Custom MIDI Args
parser.add_argument("--midi", type=str, required=True, help="Path to MIDI file.")
parser.add_argument("--force_scale", type=float, default=20.0, help="Target Force [N].")
parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of video (steps).")

# 2. Add Missing Args
def add_arg_if_missing(parser, arg_name, **kwargs):
    existing_opts = [opt for action in parser._actions for opt in action.option_strings]
    if arg_name not in existing_opts:
        parser.add_argument(arg_name, **kwargs)

add_arg_if_missing(parser, "--load_checkpoint", type=str, default="model_.*.pt", help="Checkpoint file name pattern.")
add_arg_if_missing(parser, "--load_run", type=str, default=None, help="Specific run folder name to load.")
add_arg_if_missing(parser, "--checkpoint", type=str, default=None, help="Path to specific checkpoint file.")
add_arg_if_missing(parser, "--experiment", type=str, default=None, help="Experiment folder name.")

# 3. RSL-RL Standard Args
add_arg_if_missing(parser, "--task", type=str, default=None, help="Name of the task.")
add_arg_if_missing(parser, "--num_envs", type=int, default=None, help="Number of environments.")
add_arg_if_missing(parser, "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config.")
add_arg_if_missing(parser, "--seed", type=int, default=None, help="Seed.")
add_arg_if_missing(parser, "--use_pretrained_checkpoint", action="store_true", help="Use pre-trained checkpoint.")

# 4. Add External Args
try:
    cli_args.add_rsl_rl_args(parser)
except argparse.ArgumentError:
    pass 
except Exception as e:
    print(f"[Warning] Failed to add cli_args: {e}")

AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports ---
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import porcaro_rl.tasks

# mido check
try:
    import mido
except ImportError:
    print("\n[Error] 'mido' library is not installed.")
    print("  ./isaaclab.sh -p -m pip install mido\n")
    sys.exit(1)

# ==============================================================================
# MIDI Helper Class
# ==============================================================================
class MidiInjector:
    def __init__(self, midi_path, dt, device, target_force=20.0):
        self.device = device
        self.dt = dt
        self.target_force = target_force
        
        mid = mido.MidiFile(midi_path)
        
        tempo = 500000
        for msg in mid:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
        self.bpm = mido.tempo2bpm(tempo)
        
        current_time = 0.0
        spikes = []
        for msg in mid.merged_track:
            time_delta = mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
            current_time += time_delta
            if msg.type == 'note_on' and msg.velocity > 0:
                spikes.append(current_time)
        
        self.duration_sec = current_time + 2.0
        total_steps = int(self.duration_sec / dt) + 100
        
        spike_tensor = torch.zeros((1, 1, total_steps), device=device)
        for t in spikes:
            idx = int(t / dt)
            if idx < total_steps:
                spike_tensor[0, 0, idx] = 1.0
        
        # --- 変更箇所: リズムジェネレータと波形を統一 (Super-Gaussian) ---
        width_sec = 0.035 # 0.05 -> 0.035へ変更
        sigma = width_sec / 2.0
        radius = int(width_sec / dt)
        t_vals = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32) * dt
        
        # 指数を2乗(**2)から4乗(**4)に変更し、裾野の重なりを防ぐ
        kernel = (target_force * torch.exp(-0.5 * (t_vals / sigma) ** 4)).view(1, 1, -1)
        # ------------------------------------------------------------------
        
        with torch.no_grad():
            traj = F.conv1d(spike_tensor, kernel, padding=radius)
        
        self.trajectory = traj.view(-1)
        print(f"[MIDI] Loaded {midi_path}: BPM={self.bpm:.1f}, Duration={self.duration_sec:.1f}s, Steps={total_steps}")

    def inject_to_env(self, env):
        raw_env = env.unwrapped
        
        if not hasattr(raw_env, "rhythm_generator"):
            print("[Warning] env has no 'rhythm_generator'. Skipping MIDI injection.")
            return

        gen = raw_env.rhythm_generator
        num_envs = raw_env.num_envs
        
        if hasattr(gen, "current_bpms"):
            gen.current_bpms[:] = self.bpm
        
        midi_len = self.trajectory.shape[0]
        new_traj_buffer = self.trajectory.unsqueeze(0).expand(num_envs, -1).clone()
        gen.target_trajectories = new_traj_buffer
        gen.max_steps = midi_len
        
        print("[MIDI] Injection Successful: Replaced target trajectories.")

# ==============================================================================
# Main
# ==============================================================================
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    # 1. パス解決
    checkpoint_path = getattr(args_cli, "checkpoint", None)
    load_run = getattr(args_cli, "load_run", None)
    
    experiment_name = args_cli.experiment if args_cli.experiment else agent_cfg.experiment_name
    run_dir_arg = load_run if load_run else ".*"
    
    if checkpoint_path:
        resume_path = retrieve_file_path(checkpoint_path)
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))
        resume_path = get_checkpoint_path(log_root_path, run_dir_arg, args_cli.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Loading model from: {resume_path}")

    # Config Override
    env_cfg.scene.num_envs = 1
    if hasattr(args_cli, "device") and args_cli.device:
        env_cfg.sim.device = args_cli.device
        
    env_cfg.episode_length_s = 300.0

    env_cfg.log_dir = log_dir
    if hasattr(env_cfg, "logging"):
        env_cfg.logging.enabled = True
        print("[INFO] Play mode detected: Logging enabled (force).")
    if hasattr(env_cfg, "reward_logging"):
        env_cfg.reward_logging.enabled = True
    
    # 2. 環境構築
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # Video Recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_midi"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # 3. モデルロード
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 4. MIDI データの準備 & 注入
    dt_ctrl = env.unwrapped.dt_ctrl_step
    midi_injector = MidiInjector(args_cli.midi, dt_ctrl, env.unwrapped.device, args_cli.force_scale)

    obs, _ = env.reset()
    if hasattr(policy, "reset_memory"): policy.reset_memory()
    
    midi_injector.inject_to_env(env)
    
    print("="*60)
    print(f" Sim-Verification Started (MIDI Mode)")
    print(f" Task: {args_cli.task}")
    print(f" Exp:  {experiment_name}")
    print(f" Run:  {run_dir_arg}")
    print("="*60)

    step_count = 0
    max_steps = midi_injector.trajectory.shape[0]

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                
                step_count += 1

                # --- 終了条件判定 ---
                
                # 1. MIDI曲が終わった場合
                if step_count >= max_steps:
                    print(f"Song finished successfully ({step_count} steps). Exiting...")
                    break
                
                # 2. ロボットが転倒またはリセットされた場合
                if dones.any():
                    print(f"Env reset detected at step {step_count} (Fall or Timeout). Exiting...")
                    break

    except KeyboardInterrupt:
        print("Stopped by user.")
    
    finally:
        print("[INFO] Closing environment and saving logs...")
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()