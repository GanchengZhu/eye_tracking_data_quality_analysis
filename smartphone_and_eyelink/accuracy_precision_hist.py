import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns

# tips = sns.load_dataset("tips")
eyelink_et = pd.read_csv('results_correction/eyelink/summary_eyelink.csv')
# print(cal_res)
phone_et = pd.read_csv('results_correction/phone/summary_phone.csv')

sns.palplot(sns.color_palette('RdBu_r', 4))
# colors = [
#     ["#EC817E", "#EC817E"],
#     ["#71BFB2", "#71BFB2"],
#     ["#237B9F", "#237B9F"],
# ]

# 设置线的颜色
colors = ["#c82423", "#2878b5", "#007d7d"]
# colors = [['windows blue', 'amber', ], ['greyish', 'faded green'], ['dusty purple', "#FFC208"]]
# colors = [
#     [[192,217,150], [0, 151, 92],],
#     [[255, 252, 193],[252, 195, 119]],
#     [[ 238 ,40 ,32], [249,130, 63]]
# ]
# 设置字体
font = {'family': 'Arial',
        'weight': 'bold',
        'size': 14}

# 绘图
# sns.set_style("whitegrid")  # 设置背景样式

fig, ax = plt.subplots(1, 2)  # , gridspec_kw={'height_ratios': [2, 3]})
fig.tight_layout(pad=1)
fig.set_size_inches(12, 4)

# draw ellipse
# cal_res['task'] = pd.Categorical(cal_res[1])
# cal_res['seq'] = pd.Categorical(cal_res[2])
# task = list(cal_res.task.cat.categories)
# seq = list(cal_res.seq.cat.categories)
kwargs = dict(alpha=0.75, density=False, bins=20)

for device_index, device_name in enumerate(["Portable Duo", "Phone" ]):
    for metric_index, metric_name in enumerate(["Accuracy", "Precision"]):
        if device_name == "Phone":
            et = phone_et
        else:
            et = eyelink_et
        # select data for a target pos

        ax[metric_index].hist(x=et[metric_name], color=colors[device_index],
                              label=device_name, edgecolor=colors[device_index], linewidth=2, **kwargs)

        # ax[i_s, i_t].set_title(f'Validation {s}')
        ax[metric_index].set_xlabel(f'{metric_name}', fontsize=14)
        ax[metric_index].set_ylabel("Count", fontsize=14)
        # ax[metric_index].text(1.5, 24, f'Median={np.median(et[metric_name]):.2f} cm\nMean={np.mean(et[metric_name]):.2f} cm\n80%={np.quantile(et[metric_name], 0.8):.2f} cm')
        # ax[device_index, metric_index].set_ylim([0.0, 30.0])
        # ax[device_index, metric_index].set_xlim([0, 3])

        ax[metric_index].legend(loc='upper right')
plt.tight_layout()
plt.savefig("acc_precision_hist_result.png", dpi=300)
