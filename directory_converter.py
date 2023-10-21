import os
import shutil
import pandas as pd

data_entry = pd.read_csv("sample_labels.csv")

# 画像が保存されているディレクトリのパス
image_dir = "images"  # このパスは適切なものに変更してください

# Finding Labelsからユニークなラベルのリストを取得
unique_labels = data_entry['Finding Labels'].str.split('|').apply(lambda x: x[0]).unique()

# ラベルごとのサブディレクトリを作成
for label in unique_labels:
    label_dir = os.path.join(image_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

# 各画像を対応するサブディレクトリに移動
for index, row in data_entry.iterrows():
    image_name = row['Image Index']
    label = row['Finding Labels'].split('|')[0]
    #import pdb;pdb.set_trace()
    source_path = os.path.join(image_dir, image_name)
    destination_path = os.path.join(image_dir, label, image_name)
    if os.path.exists(source_path):
        shutil.copy(source_path, destination_path)

# 処理が完了したかどうかのフラグを返す
completed = True
completed
