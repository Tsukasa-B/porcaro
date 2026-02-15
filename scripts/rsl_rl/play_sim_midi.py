"""
Play script with MIDI Injection (No Env Modification Ver.)
既存の環境コードを変更せず、外部からMIDIデータを注入してテストします。

Usage:
  python scripts/rsl_rl/play_sim_midi.py \
  --task Template-Porcaro-Direct-ModelB \
  --load_run [YOUR_RUN_NAME] \
  --midi songs/test_single4_bpm60.mid \
  --video
"""

import argparse
import sys
import os
import torch
import torch.nn.functional as F
import mido
import numpy as np
import gymnasium as gym

from isaaclab.app import AppLauncher
import cli_args

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Play RL agent with MIDI Input (Injection Mode).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of video (steps).")
parser.add_argument("--midi", type=str, required=True, help="Path to MIDI file.")
parser.add_argument("--force_scale", type=float, default=50.0, help="Target Force [N].")

cli_args.add_rsl_rl_args(parser)
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

# ==============================================================================
# MIDI Helper Class (Internal)
# ==============================================================================
class MidiInjector:
    def __init__(self, midi_path, dt, device, target_force=50.0):
        self.device = device
        self.dt = dt
        self.target_force = target_force
        
        # MIDI読み込みと軌道生成
        mid = mido.MidiFile(midi_path)
        
        # テンポ解析
        tempo = 500000
        for msg in mid:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
        self.bpm = mido.tempo2bpm(tempo)
        
        # ノートイベント抽出
        current_time = 0.0
        spikes = []
        for msg in mid.merged_track:
            time_delta = mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
            current_time += time_delta
            if msg.type == 'note_on' and msg.velocity > 0:
                spikes.append(current_time)
        
        self.duration_sec = current_time + 2.0
        total_steps = int(self.duration_sec / dt) + 100
        
        # 軌道生成 (Conv1d)
        spike_tensor = torch.zeros((1, 1, total_steps), device=device)
        for t in spikes:
            idx = int(t / dt)
            if idx < total_steps:
                spike_tensor[0, 0, idx] = 1.0
        
        width_sec = 0.05
        sigma = width_sec / 2.0
        radius = int(width_sec / dt)
        t_vals = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32) * dt
        kernel = (target_force * torch.exp(-0.5 * (t_vals / sigma) ** 2)).view(1, 1, -1)
        
        with torch.no_grad():
            traj = F.conv1d(spike_tensor, kernel, padding=radius)
        
        self.trajectory = traj.view(-1) # [TotalSteps]
        print(f"[MIDI] Loaded {midi_path}: BPM={self.bpm:.1f}, Duration={self.duration_sec:.1f}s, Steps={total_steps}")

    def inject_to_env(self, env):
        """
        環境内のRhythmGeneratorのデータを強制的に書き換える
        """
        # env.unwrapped で生の PorcaroRLEnv を取得
        raw_env = env.unwrapped
        
        if not hasattr(raw_env, "rhythm_generator"):
            print("[Error] env has no rhythm_generator!")
            return

        gen = raw_env.rhythm_generator
        num_envs = raw_env.num_envs
        max_steps_env = gen.max_steps # Env側で確保されているバッファサイズ
        
        # 1. BPMの上書き
        gen.current_bpms[:] = self.bpm
        
        # 2. ターゲット軌道の上書き
        # MIDIデータが長すぎる場合、Envのバッファサイズに合わせてカットするか、
        # Env側の仕組みを無視して参照先をすげ替える必要がある。
        # ここでは「参照先すげ替え」を行う (Pythonならではの荒業)
        
        # 新しい巨大なバッファを作成 [num_envs, midi_len]
        midi_len = self.trajectory.shape[0]
        new_traj_buffer = self.trajectory.unsqueeze(0).expand(num_envs, -1).clone()
        
        # ★重要: クラスのインスタンス変数を丸ごと差し替える
        gen.target_trajectories = new_traj_buffer
        gen.max_steps = midi_len # カウンタ上限も書き換え
        
        # エピソード長の上書き (タイムアウト防止)
        if hasattr(raw_env, "max_episode_length"):
            raw_env.max_episode_length = midi_len + 100
            
        # エピソード長バッファ(episode_duration_steps)も更新
        if hasattr(raw_env, "episode_duration_steps"):
            raw_env.episode_duration_steps[:] = midi_len

        print("[MIDI] Injection Successful: Replaced target trajectories.")


# ==============================================================================
# Main
# ==============================================================================
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    # 1. パス解決 & Config設定
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # Config Override
    env_cfg.scene.num_envs = 1 # MIDI再生は1環境で十分
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"
    
    # ★重要: エピソード長を非常に長く設定し、勝手にリセットされないようにする
    # (初期設定値を上書き)
    env_cfg.episode_length_s = 300.0 # 5分あれば十分
    
    # 2. 環境構築
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # Video
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_midi"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length, # ここで録画長さを指定
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # 3. モデルロード
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 4. MIDI データの準備
    dt_ctrl = env.unwrapped.dt_ctrl_step # PorcaroRLEnvから取得
    midi_injector = MidiInjector(args_cli.midi, dt_ctrl, env.unwrapped.device, args_cli.force_scale)

    # 5. シミュレーション開始
    obs, _ = env.reset()
    if hasattr(policy, "reset_memory"): policy.reset_memory()

    # ★★★ ここで注入！ ★★★
    # reset()直後に行うことで、初期化されたランダムパターンをMIDIデータで上書きする
    midi_injector.inject_to_env(env)
    
    print("="*60)
    print(f" Sim-Verification Started (MIDI Mode)")
    print("="*60)

    step_count = 0
    max_steps = midi_injector.trajectory.shape[0]

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, terminated, truncated, _ = env.step(actions)
            
            step_count += 1
            
            # 再生終了判定
            if step_count >= max_steps:
                print("Song finished.")
                break
            
            # もし環境側のタイムアウトでリセットされたら、再度注入が必要
            # (ただし max_episode_length を大きくしているので基本起きないはず)
            if terminated.any() or truncated.any():
                print("Env reset detected. Re-injecting MIDI...")
                midi_injector.inject_to_env(env)
                step_count = 0

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()