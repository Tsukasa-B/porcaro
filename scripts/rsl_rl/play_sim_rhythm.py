"""
Script to play a checkpoint with SPECIFIC rhythm injection (Sim Verification).
Target: Isaac Lab / RSL-RL
Matches logic with: run_deploy_v4.py

Usage:
  python scripts/rsl_rl/play_sim_rhythm.py --load_run [RunName] --pattern double --bpm 120
"""

import argparse
import sys
import os
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

# local imports
import cli_args

# --- Argument Parser Setup ---
parser = argparse.ArgumentParser(description="Play RL agent with Controlled Rhythm Input.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
parser.add_argument("--video_length", type=int, default=400, help="Length of video.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default="Isaac-Porcaro-Direct-v0", help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config.")
parser.add_argument("--seed", type=int, default=None, help="Seed.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use pretrained ckpt.")

# ★ デプロイスクリプトと共通の引数
parser.add_argument("--bpm", type=float, default=60.0, help="Target BPM (e.g. 60, 120).")
parser.add_argument("--pattern", type=str, default="single_4", 
                    choices=["single_4", "single_8", "double", "paradiddle", "upbeat", "clave"],
                    help="Rhythm pattern to test.")

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

# --- Main Logic ---
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    # 1. パス解決
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    
    log_dir = os.path.dirname(resume_path)

    # 2. 環境設定のオーバーライド (ここが重要)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else "cuda:0"
    
    # ログ有効化
    if hasattr(env_cfg, "logging"):
        env_cfg.logging.enabled = True 

    # ★ RhythmGeneratorを「テストモード」にする設定
    # ※ PorcaroRLEnvCfg にこれらのフィールドがある前提ですが、
    #    もしなければ後述の `env.unwrapped` で直接注入します。
    if hasattr(env_cfg, "use_simple_rhythm"):
        env_cfg.use_simple_rhythm = True
        env_cfg.simple_rhythm_mode = args_cli.pattern
        env_cfg.simple_rhythm_bpm = args_cli.bpm
        print(f"[Config] Config Overridden: Pattern={args_cli.pattern}, BPM={args_cli.bpm}")

    # 3. 環境構築
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # Video録画設定
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_sim_rhythm"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # 4. エージェントロード
    print(f"[INFO]: Loading model from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 5. 直接注入 (念の為の二重対策)
    # Config経由で設定されなかった場合、直接内部のRhythmGeneratorを叩く
    try:
        unwrapped_env = env.unwrapped
        if hasattr(unwrapped_env, "rhythm_generator"):
            print("[INFO] Setting RhythmGenerator Test Mode directly...")
            unwrapped_env.rhythm_generator.set_test_mode(
                enabled=True, 
                bpm=args_cli.bpm, 
                pattern=args_cli.pattern
            )
            # 全環境のリズムをリセットして反映
            unwrapped_env.rhythm_generator.reset(
                torch.arange(unwrapped_env.num_envs, device=unwrapped_env.device)
            )
    except Exception as e:
        print(f"[WARNING] Could not set rhythm generator directly: {e}")

    # --- Simulation Loop ---
    obs, _ = env.reset()
    
    # RNNリセット
    if hasattr(policy, "reset_memory"):
        policy.reset_memory()
    
    print("="*60)
    print(f" Sim-Verification Started")
    print(f" Mode: {args_cli.pattern} | BPM: {args_cli.bpm}")
    print("="*60)

    while simulation_app.is_running():
        with torch.inference_mode():
            # 推論
            actions = policy(obs)
            
            # ステップ実行 (環境がConfig通りのリズムを生成してくれる)
            obs, _, _, _ = env.step(actions)
            
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()