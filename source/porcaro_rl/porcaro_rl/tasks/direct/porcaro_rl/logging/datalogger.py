# data_logger.py (新規作成)
from __future__ import annotations
import csv
import os
from typing import Sequence
import atexit

class DataLogger:
    """
    シミュレーションデータをCSVファイルに蓄積・保存するクラス。
    並列環境のうち、インデックス0の環境のデータのみを記録することを想定。
    """
    def __init__(self, filepath: str, headers: Sequence[str]):
        self.filepath = filepath
        self.headers = list(headers)
        self.buffer: list[list[float]] = []
        
        # 起動時にヘッダーを書き込む（既存ファイルは上書き）
        self._write_header()

        atexit.register(self.save)

    def _write_header(self):
        """CSVファイルにヘッダーを書き込む（既存ファイルは上書き）"""
        try:
            with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
            print(f"[DataLogger] ヘッダーを {self.filepath} に書き込みました。")
        except OSError as e:
            print(f"[DataLogger] ERROR: ヘッダーの書き込みに失敗しました。 {e}")

    def add_step(self, data_row: Sequence[float]):
        """
        1ステップ分のデータ（Pythonのfloatリスト）をバッファに追加する。
        data_row の要素数は headers と一致している必要がある。
        """
        if len(data_row) != len(self.headers):
            print(f"[DataLogger] WARN: データ長({len(data_row)})がヘッダー長({len(self.headers)})と一致しません。スキップします。")
            return
        self.buffer.append(list(data_row))

    def add_data_batch(self, rows: list[list]):
        """複数行のデータをバッファに一括で追加します"""
        # extend を使うと、リストの全要素が一度に追加される
        self.buffer.extend(rows)

    def save(self):
        """バッファに蓄積されたデータをCSVファイルに追記する"""
        if not self.buffer:
            # バッファが空でも、それが分かるようにログを出す
            print("[DataLogger] バッファが空です。書き込みをスキップしました。")
            return
        
        try:
            # 'a' (追記) モードでファイルを開く
            with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(self.buffer)
            
            num_rows = len(self.buffer)
            self.buffer.clear()
            # --- 変更点： print のコメントアウトを解除 ---
            print(f"[DataLogger] {num_rows} 件のデータを {self.filepath} に追記しました。")
            
        except OSError as e:
            print(f"[DataLogger] ERROR: CSVファイルへの書き込みに失敗しました。 {e}")
            
    def clear_buffer(self):
        """バッファをクリアする"""
        self.buffer.clear()