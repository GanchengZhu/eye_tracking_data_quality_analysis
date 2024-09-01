import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

max_x = 1046.4656
max_y = 2214.684
min_x = 83.12728
min_y = 87.28937

acc_min_value = 0
acc_max_value = 3

pre_min_value = 0
pre_max_value = 0.5

n_row = 10
n_col = 5

col_stride = (max_x - min_x) / n_col
row_stride = (max_y - min_y) / n_row

# 设置字体
font = {'family': 'Arial',
        'weight': 'bold',
        'size': 14}
plt.rc('font', **font)


def get_index_x(x):
    x = covertX(x)
    for i in range(n_col):
        low = (min_x + i * col_stride)
        high = (min_x + (i + 1) * col_stride)
        if low <= x <= high:
            return i
        # if i == n_col - 1 and x <= max_x:
        #     return i
    raise Exception("Not find index of ", x)


def get_index_y(y):
    y = covertY(y)
    for i in range(n_row):
        low = (min_y + i * row_stride)
        high = (min_y + (i + 1) * row_stride)
        if low <= y <= high:
            return i
        # if i == n_row - 1 and y <= max_y:
        #     return i
    raise Exception("Not find index of ", y)


def covertX(x):
    x /= 7.4
    x *= 1080
    return x


def covertY(y):
    y /= 15.34293
    y *= 2249
    return y


def init_heatmap_data(heatmap_data):
    for i in range(n_row):
        row = []
        for j in range(n_col):
            row.append([])
        heatmap_data.append(row)


def mean_heatmap(heatmap_data):
    for i in range(n_row):
        for j in range(n_col):
            heatmap_data[i][j] = np.mean(heatmap_data[i][j])


eyelink_data_source = glob.glob("results_correction/eyelink/subjects/eyelink_*.csv")
phone_data_source = glob.glob("results_correction/phone/subjects/phone_*.csv")

eyelink_acc_heatmap_data = []
eyelink_pre_heatmap_data = []
phone_acc_heatmap_data = []
phone_pre_heatmap_data = []

init_heatmap_data(eyelink_acc_heatmap_data)
init_heatmap_data(eyelink_pre_heatmap_data)
init_heatmap_data(phone_pre_heatmap_data)
init_heatmap_data(phone_acc_heatmap_data)

# eye_link
for csv_data_path in eyelink_data_source:
    subject_df = pd.read_csv(csv_data_path)
    for row in subject_df.iterrows():
        row = row[1]
        # print(row)
        # print(covertX(row.gt_x), " ", covertY(row.gt_y))
        index_x = get_index_x(row.gt_x)
        index_y = get_index_y(row.gt_y)
        # print(index_x, " ", index_y)
        # print(" ")
        if not np.isnan(row.accuracy):
            eyelink_acc_heatmap_data[index_y][index_x].append(row.accuracy)
        if not np.isnan(row.precision):
            eyelink_pre_heatmap_data[index_y][index_x].append(row.precision)

# phone
for csv_data_path in phone_data_source:
    subject_df = pd.read_csv(csv_data_path)
    for row in subject_df.iterrows():
        row = row[1]
        # print(row)
        # print(covertX(row.gt_x), " ", covertY(row.gt_y))
        index_x = get_index_x(row.gt_x)
        index_y = get_index_y(row.gt_y)
        # print(index_x, " ", index_y)
        # print(" ")
        if not np.isnan(row.accuracy):
            phone_acc_heatmap_data[index_y][index_x].append(row.accuracy)
        if not np.isnan(row.precision):
            phone_pre_heatmap_data[index_y][index_x].append(row.precision)

mean_heatmap(eyelink_acc_heatmap_data)
mean_heatmap(eyelink_pre_heatmap_data)
mean_heatmap(phone_pre_heatmap_data)
mean_heatmap(phone_acc_heatmap_data)

# acc
# 创建热力图
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))
eyelink_heatmap = ax[1].imshow(eyelink_acc_heatmap_data, cmap='Blues', vmin=acc_min_value, vmax=acc_max_value)  # cmap参数指定色彩图
phone_heatmap = ax[0].imshow(phone_acc_heatmap_data, cmap='Blues', vmin=acc_min_value, vmax=acc_max_value)  # cmap参数指定色彩图
# 隐藏坐标轴
ax[0].set_xticks(ticks=np.arange(n_col), labels=[f'{int((i + 0.5) * row_stride + min_x)}' for i in range(n_col)])
ax[0].set_yticks(ticks=np.arange(n_row), labels=[f'{int((i + 0.5) * row_stride + min_y)}' for i in range(n_row)])
ax[1].set_xticks(ticks=np.arange(n_col), labels=[f'{int((i + 0.5) * row_stride + min_x)}' for i in range(n_col)])
ax[1].set_yticks(ticks=np.arange(n_row), labels=[f'{int((i + 0.5) * row_stride + min_y)}' for i in range(n_row)])

# 添加色条以显示色彩映射
fig.colorbar(eyelink_heatmap, ax=ax[1])
fig.colorbar(phone_heatmap, ax=ax[0])
# 显示图形
plt.tight_layout()
plt.savefig("./figures/heatmap_acc.jpg", dpi=300)

# pre
# 创建热力图
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))
eyelink_heatmap = ax[1].imshow(eyelink_pre_heatmap_data, cmap='Reds', vmin=pre_min_value, vmax=pre_max_value)  # cmap参数指定色彩图
phone_heatmap = ax[0].imshow(phone_pre_heatmap_data, cmap='Reds', vmin=pre_min_value, vmax=pre_max_value)  # cmap参数指定色彩图
# 隐藏坐标轴
ax[0].set_xticks(ticks=np.arange(n_col), labels=[f'{int((i + 0.5) * row_stride + min_x)}' for i in range(n_col)])
ax[0].set_yticks(ticks=np.arange(n_row), labels=[f'{int((i + 0.5) * row_stride + min_y)}' for i in range(n_row)])
ax[1].set_xticks(ticks=np.arange(n_col), labels=[f'{int((i + 0.5) * row_stride + min_x)}' for i in range(n_col)])
ax[1].set_yticks(ticks=np.arange(n_row), labels=[f'{int((i + 0.5) * row_stride + min_y)}' for i in range(n_row)])

# 添加色条以显示色彩映射
fig.colorbar(eyelink_heatmap, ax=ax[1])
fig.colorbar(phone_heatmap, ax=ax[0])

# 显示图形
plt.tight_layout()
plt.savefig("./figures/heatmap_pre.jpg", dpi=300)
