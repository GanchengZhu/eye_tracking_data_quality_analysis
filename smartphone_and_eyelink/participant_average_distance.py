import glob
import os
import numpy as np
import pandas as pd

file_lists = glob.glob("data_quality_results/phone/subjects/phone_*.csv")
SKIP_IDS = {6, 23, 31, 32}

distance_list = []

for f in file_lists:
    # 提取文件 ID
    file_id = int(os.path.basename(f).split("_")[1].split(".")[0])
    if file_id in SKIP_IDS:
        continue

    df = pd.read_csv(f)

    # 确保 'distance' 列存在
    if 'distance' in df.columns:
        distance_list.extend(df['distance'].dropna().values)  # 过滤 NaN
    else:
        print(f"Warning: 'distance' column not found in {f}")

# 计算均值和标准差
if distance_list:
    print(np.mean(distance_list))
    print(np.std(distance_list))
else:
    print("No valid distance data found.")
