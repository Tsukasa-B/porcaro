import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
import os

def create_porcaro_midi(filename, pattern_name, bpm):
    """
    Porcaroプロジェクト (rhythm_generator.py) の定義に完全準拠したMIDIを生成する。
    構成: 4小節 (Bar 0: 休符, Bar 1-3: 指定パターン)
    
    【片腕ロボット用リズム定義】
    - single_4: [0, 8] (2分音符相当。1小節に2回のみ)
    - single_8: [0, 4, 8, 12] (4分音符相当。1小節に4回)
    """
    
    # 1. Porcaro Rhythm Definitions (User Defined)
    rudiments = {
        # 表打ち (片腕用間引き: 2分音符) - [0, 8]
        "single_4":  [0, 8],
        # 8ビート (片腕用間引き: 4分音符) - [0, 4, 8, 12]
        "single_8":  [0, 4, 8, 12],
        # ダブルストローク (RRLL...) - バネ性を活かしたリバウンド動作 [0, 1, 4, 5...]
        "double":    [0, 1, 4, 5, 8, 9, 12, 13],
        # 休符
        "rest":      []
    }

    if pattern_name not in rudiments:
        print(f"[Error] Pattern '{pattern_name}' not found.")
        return

    # 2. MIDI Setup
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    # テンポ設定
    tempo = mido.bpm2tempo(bpm)
    track.append(MetaMessage('set_tempo', tempo=tempo))
    
    # 分解能 (Ticks per Beat)
    ticks_per_beat = 480
    mid.ticks_per_beat = ticks_per_beat
    
    # 16分音符1つあたりのTick数 (4分音符 / 4)
    tick_16th = int(ticks_per_beat / 4)
    
    # 3. ノート生成ループ (4小節固定)
    # Bar 0: Rest (Count-in)
    # Bar 1-3: Pattern
    bar_patterns = ["rest", pattern_name, pattern_name, pattern_name]
    
    print(f"Creating {filename} (BPM: {bpm})")
    print(f"Structure: {bar_patterns}")
    
    # 時間管理用変数 (絶対時間)
    last_tick = 0 
    
    # ドラムノート番号 (Snare = 38)
    note_num = 38
    velocity = 100

    for bar_idx, p_name in enumerate(bar_patterns):
        hit_indices = rudiments[p_name]
        
        for grid_idx in range(16):
            if grid_idx in hit_indices:
                # 打撃すべき絶対時間 (Absolute Tick)
                strike_tick = (bar_idx * 16 + grid_idx) * tick_16th
                
                # --- Note ON ---
                # 前のイベント(last_tick)からの差分(delta)を計算
                delta_on = strike_tick - last_tick
                
                # deltaが負になる(計算ミス)を防ぐガード
                if delta_on < 0: delta_on = 0
                
                track.append(Message('note_on', note=note_num, velocity=velocity, time=delta_on))
                
                # --- Note OFF ---
                # 10ticks後に切る (音の強弱変化を防ぐため短く)
                duration = 10 
                track.append(Message('note_off', note=note_num, velocity=0, time=duration))
                
                # 最後にイベントを書き込んだ時間を更新
                last_tick = strike_tick + duration
            
    # 4. トラックの終了 (End of Track)
    # 最後の小節のお尻まで時間を進める
    total_ticks = 4 * 16 * tick_16th # 4小節分の総Tick数
    
    remaining = total_ticks - last_tick
    if remaining > 0:
        # 空白時間を埋めるためのダミーNoteOff
        track.append(Message('note_off', note=note_num, velocity=0, time=remaining))
        
    mid.save(filename)
    print(f"Saved: {filename}\n")

if __name__ == "__main__":
    os.makedirs("songs", exist_ok=True)
    
    # --- 実験用MIDIセット生成 ---
    
    # 1. 基礎動作 (BPM 60) - 片腕負荷低減版
    # single_4 (2打/小節)
    create_porcaro_midi("songs/test_single4_bpm60.mid", "single_4", 60)
    
    # 2. ダブルストローク (BPM 60)
    create_porcaro_midi("songs/test_double_bpm60.mid", "double", 60)
    
    # 3. 中速動作 (BPM 120)
    # single_8 (4打/小節)
    create_porcaro_midi("songs/test_single8_bpm120.mid", "single_8", 120)
    create_porcaro_midi("songs/test_double_bpm120.mid", "double", 120)

    # 4. 高速動作 (BPM 160)
    create_porcaro_midi("songs/test_single8_bpm160.mid", "single_8", 160)
    create_porcaro_midi("songs/test_double_bpm160.mid", "double", 160)