import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

# data = pd.read_excel("filtered_data/draw.xlsx")

font = {'family': 'Arial',
        'size': 20,}
plt.rc('font', **font)

# 设置时间轴
# time = np.arange(len(data.x_filtered_one)) * 100 / 3

# 创建子图
# fig, axs = plt.subplots(2, 2, figsize=(12, 8))



# 设置线的颜色
colors = ["#c82423", "#2878b5",  "#007d7d"]

# # 用循环填充子图数据和标题
# for i, ax in enumerate(axs.flat):
#     ax.set_xlabel('Time (ms)', fontsize=16)
#     ax.set_ylabel('Centimeters', fontsize=16)
#
#     if i == 0:  # Groundtruth
#         ax.plot(time, data.x_gt, label='Horizontal signal', color=colors[0])
#         ax.plot(time, data.y_gt, label='Vertical signal', color=colors[1])
#         ax.legend(frameon=True, fontsize=12)
#     elif i == 1:  # Raw Output (示例数据)
#         ax.plot(time, data.x_pre, label='Horizontal signal', color=colors[0])
#         ax.plot(time, data.y_pre, label='Vertical signal', color=colors[1])
#         ax.legend(frameon=True, fontsize=12)
#     elif i == 2:  # Heuristic filter (示例数据)
#         ax.plot(time, data.x_filtered_heu, label='Horizontal signal', color=colors[0])
#         ax.plot(time, data.y_filtered_heu, label='Vertical signal', color=colors[1])
#         ax.legend(frameon=True, fontsize=12)
#     elif i == 3:  # One Euro filter (示例数据)
#         ax.plot(time, data.x_filtered_one, label='Horizontal signal', color=colors[0])
#         ax.plot(time, data.y_filtered_one, label='Vertical signal', color=colors[1])
#         ax.legend(frameon=True, fontsize=12)
#
#     # 设置子图标题
#     ax.set_title(titles[i], fontsize=16, y=-.3)


data_source = "right"
for i in range(1, 33):
    eyelink_data_path = glob.glob("dataset/eyelink/%02d/*.csv" % i)
    print(eyelink_data_path)
    assert len(eyelink_data_path) == 1
    eyelink_data_path = eyelink_data_path[0]

    phone_data_path = glob.glob("dataset/phone/%02d/*.txt" % i)
    assert len(phone_data_path) == 1
    phone_data_path = phone_data_path[0]
    print(phone_data_path)

    phone_df = pd.read_csv(phone_data_path)
    eyelink_df = pd.read_csv(eyelink_data_path)

    phone_t = phone_df.timestamp - phone_df.timestamp[0]
    phone_t = phone_t / 1E6
    print(len(phone_t))

    eyelink_df = eyelink_df.iloc[:, ]
    eyelink_t = eyelink_df.phone_timestamp - phone_df.timestamp[0]
    eyelink_t = eyelink_t / 1E6

    scale_x = 1080 / 332
    scale_y = 2249 / 692

    if data_source == "right":
        eyelink_x = eyelink_df.right_x * scale_x
        eyelink_y = eyelink_df.right_y * scale_y
    else:
        eyelink_x = eyelink_df.left_x * scale_x
        eyelink_y = eyelink_df.left_y * scale_y

    eyelink_x[eyelink_x < 0] = np.nan
    eyelink_y[eyelink_y < 0] = np.nan

    eyelink_x[eyelink_x > 1080] = np.nan
    eyelink_y[eyelink_y > 2249] = np.nan

    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])  # 上面的图占2个单位高度，下面的图占1个单位高度

    # 创建子图，并设置共享y轴
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    ax1.set_xlabel('Time (ms)', fontsize=20)
    ax1.set_ylabel('Pixels', fontsize=20)

    ax2.set_xlabel('Time (ms)', fontsize=20)
    ax2.set_ylabel('Pixels', fontsize=20)

    ax1.plot(phone_t, phone_df.filteredX, color=colors[0], label='Phone', )
    ax1.plot(phone_t, phone_df.gtX, color=colors[2], label='Ground Truth', )
    ax1.plot(eyelink_t, eyelink_x, color=colors[1], label='Portable Duo', )
    ax1.set_title("Horizontal signal", fontsize=20, y=-0.5)
    # ax1.set_ylim(0, 1080)
    # ax1.legend()
    # ax1.set_title("%02d" % i)
    ax2.plot(phone_t, phone_df.filteredY, color=colors[0], label='Phone', )
    ax2.plot(phone_t, phone_df.gtY, color=colors[2], label='Ground Truth', )
    ax2.plot(eyelink_t, eyelink_y, color=colors[1], label='Portable Duo', )
    ax2.set_title('Vertical signal', fontsize=20, y=-.25)
    # ax2.set_ylim(0, 2249)
    # ax2.legend()
    ax1.set_ylim(0, 1080)
    ax2.set_ylim(0, 2249)

    ax1.legend(frameon=True, fontsize=12)
    ax2.legend(frameon=True, fontsize=12)

    # 调整子图间距
    plt.subplots_adjust(hspace=0.5)
    # 调整子图布局
    plt.tight_layout()



    if data_source == "right":
        fig_save_dir = 'figures/eyelink_right_eye'
    else:
        fig_save_dir = 'figures/eyelink_left_eye'

    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)

    fig.savefig(os.path.join(fig_save_dir, "sub_%02d" % i), dpi=300)



#
# # 显示图形
# plt.show()
