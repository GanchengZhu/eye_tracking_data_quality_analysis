import glob
import os.path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
import numpy as np


plt.rcParams['font.family'] = ['Arial']
plt.rcParams['font.size'] = 18

arg = sys.argv
data_source = arg[1]


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

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])  # 上面的图占2个单位高度，下面的图占1个单位高度

    # 创建子图，并设置共享y轴
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    ax1.plot(phone_t, phone_df.filteredX, color='k', label='Phone', )
    ax1.plot(phone_t, phone_df.gtX, color='r', label='Ground Truth', )
    ax1.plot(eyelink_t, eyelink_x, color='b', label='Portable Duo', )
    # ax1.set_ylim(0, 1080)
    ax1.legend()
    ax1.set_title("%02d" % i)
    ax2.plot(phone_t, phone_df.filteredY, color='k', label='Phone', )
    ax2.plot(phone_t, phone_df.gtY, color='r', label='Ground Truth', )
    ax2.plot(eyelink_t, eyelink_y, color='b', label='Portable Duo', )
    # ax2.set_ylim(0, 2249)
    ax2.legend()
    ax1.set_ylim(0, 1080)
    ax2.set_ylim(0, 2249)

    if data_source == "right":
        fig_save_dir = 'figures/eyelink_right_eye'
    else:
        fig_save_dir = 'figures/eyelink_left_eye'

    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)

    fig.savefig(os.path.join(fig_save_dir, "sub_%02d" % i), dpi=300)
