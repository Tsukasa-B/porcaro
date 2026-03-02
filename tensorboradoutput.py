import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def export_tensorboard_data(log_dir, output_dir):
    """
    TensorBoardのログファイルからスカラーデータを読み込み、
    タグごとにCSVファイルとして一括出力します。
    """
    # 出力先のフォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # ログディレクトリ内のファイルを探索
    for root, _, files in os.walk(log_dir):
        for file in files:
            if "events.out.tfevents" in file:
                event_path = os.path.join(root, file)
                print(f"読み込み中: {event_path}")
                
                # イベントデータを読み込む
                acc = EventAccumulator(event_path)
                acc.Reload()
                
                # スカラー（折れ線グラフなどの数値）データを抽出
                if 'scalars' in acc.Tags():
                    for tag in acc.Tags()['scalars']:
                        events = acc.Scalars(tag)
                        
                        # Pandas DataFrameに変換
                        df = pd.DataFrame(
                            [(e.wall_time, e.step, e.value) for e in events],
                            columns=['wall_time', 'step', 'value']
                        )
                        
                        # 保存時のエラーを防ぐため、タグ名のスラッシュをアンダースコアに置換
                        safe_tag = tag.replace("/", "_").replace("\\", "_")
                        out_path = os.path.join(output_dir, f"{safe_tag}.csv")
                        
                        # CSVファイルとして保存
                        df.to_csv(out_path, index=False)
                        print(f"  保存完了: {out_path}")

# ==========================================
# 変更が必要なポイント
# ==========================================
# 以下の2行を、ご自身の環境に合わせて変更してください。

# 1. TensorBoardに読み込ませているログフォルダのパスを指定
log_directory = "./logs/rsl_rl/porcaro_rslrl_mlp_modelB_DR_lookahead05" 

# 2. CSVを保存したいフォルダの名前を指定（自動で作成されます）
output_directory = "./exported_csv_data/porcaro_rslrl_mlp_modelB_DR_lookahead05" 

# ==========================================

# 実行
export_tensorboard_data(log_directory, output_directory)