import os
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns

# 读取数据
eyelink_et = pd.read_csv('data_quality_results/eyelink/summary_eyelink.csv')
phone_et = pd.read_csv('data_quality_results/phone/summary_phone.csv')

# 设置颜色
colors = ["#c82423", "#2878b5", "#007d7d"]

# 设置字体
plt.rcParams.update({'font.family': 'Arial',
                     # 'font.weight': 'bold',
                     'font.size': 14,
                     'axes.labelsize': 14,
                     'axes.labelweight': 'bold'})

# 创建子图
fig, ax = plt.subplots(1, 3)
fig.tight_layout(pad=1)
fig.set_size_inches(12.6, 4)

kwargs = dict(alpha=0.75, density=False, bins=20)

# 子图标签
subplot_labels = ["(A)", "(B)", "(C)"]

for device_index, device_name in enumerate(["Portable Duo", "Smartphone"]):
    for metric_index, metric_name in enumerate(["Accuracy", "Precision", "ME"]):
        et = phone_et if device_name == "Smartphone" else eyelink_et

        # 绘制直方图
        ax[metric_index].hist(x=et[metric_name], color=colors[device_index],
                              label=device_name, edgecolor=colors[device_index], linewidth=2, **kwargs)

        if metric_name != "ME":
            ax[metric_index].set_xlabel(f'{metric_name} (°)')
        else:
            ax[metric_index].set_xlabel(f'{metric_name} (cm)')

        ax[metric_index].set_ylabel("Count")

        # 在左上角添加 (A), (B), (C)
        ax[metric_index].text(-0.2, 1.05, subplot_labels[metric_index], transform=ax[metric_index].transAxes,
                              fontsize=20, va='top', ha='left')

        legend_props = {'weight': 'bold'}  # 'weight' or 'fontweight'
        ax[metric_index].legend(loc='upper right', prop=legend_props)

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/Fig 4.png", dpi=300)
plt.show()
