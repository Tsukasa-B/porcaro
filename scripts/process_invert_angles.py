import pandas as pd
import os
import sys

def main():
    # 1. ファイルパス設定
    # ※ jetson_projectフォルダ内の対象CSVを指定してください
    input_csv_name = "data_exp4_validation_seq_.csv"
    output_csv_name = "data_exp4_validation_inverted.csv"

    # パス解決 (scriptsフォルダから実行することを想定)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # external_data/jetson_project 以下にあると仮定
    # 見つからない場合は適宜パスを修正してください
    input_path = os.path.join(project_root, "external_data", "jetson_project", input_csv_name)
    output_path = os.path.join(project_root, "external_data", "jetson_project", output_csv_name)

    if not os.path.exists(input_path):
        # 簡易的なパス探索（カレントディレクトリ等）
        if os.path.exists(input_csv_name):
            input_path = input_csv_name
            output_path = output_csv_name
        else:
            print(f"[Error] Input file not found: {input_path}")
            return

    # 2. 読み込み & 反転処理
    print(f"[Info] Loading: {input_path}")
    df = pd.read_csv(input_path)

    # 角度カラムを反転 (meas_ang_wrist, meas_ang_hand)
    # ※ 元がマイナスで、プラスにしたい場合は -1 をかけます
    target_cols = ['meas_ang_wrist', 'meas_ang_hand']
    
    for col in target_cols:
        if col in df.columns:
            print(f"[Proc] Inverting sign for {col}...")
            df[col] = df[col] * -1.0
        else:
            print(f"[Warn] Column '{col}' not found. Skipping.")

    # 3. 保存
    df.to_csv(output_path, index=False)
    print(f"[Success] Saved inverted data to: {output_path}")

if __name__ == "__main__":
    main()