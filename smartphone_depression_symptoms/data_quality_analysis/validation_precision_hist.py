import re

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

def add_space_between_words(text):
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

# tips = sns.load_dataset("tips")
cal_res = pd.read_csv('./summary/validation_summary.csv', engine='python')
print(cal_res)
sns.palplot(sns.color_palette('RdBu_r', 9))
colors = [
    ["#EC817E", "#EC817E"],
    ["#71BFB2", "#71BFB2"],
    ["#237B9F", "#237B9F"],
]
# colors = [['windows blue', 'amber', ], ['greyish', 'faded green'], ['dusty purple', "#FFC208"]]
# colors = cal_res[
#     [[192,217,150], [0, 151, 92],],
#     [[255, 252, 193],[252, 195, 119]],
#     [[ 238 ,40 ,32], [249,130, 63]]
# ]
# 设置字体
font = {'family': 'Times New Roman',
        'size': 12}
plt.rc('font', **font)

# 绘图
# sns.set_style("whitegrid")  # 设置背景样式

fig, ax = plt.subplots(2, 3)  # , gridspec_kw={'height_ratios': [2, 3]})
fig.tight_layout(pad=1)
fig.set_size_inches(12, 8)

# draw ellipse
# ['task'] = pd.Categorical(cal_res[1])
# cal_res['validation'] = pd.Categorical(cal_res[2])
task = list(pd.Categorical(cal_res.task).categories)
seq = list(pd.Categorical(cal_res.validation).categories)

print(task)
print(seq)

for t in task:
    i_t = task.index(t)
    # 8 columns
    for s in seq:
        i_s = seq.index(s)
        # select data for a target pos
        _d2plot = cal_res[(cal_res.task == t) & (cal_res.validation == s)]
        sns.histplot(data=_d2plot.precision, kde=False, ax=ax[i_s, i_t], color=colors[i_t][i_s], alpha=1)
        # ax[i_s, i_t].set_title(f'Validation {s}')
        ax[i_s, i_t].set_xlabel(f'{add_space_between_words(t)} - {"Pre" if s==0 else "Post"}-task evaluation', fontsize=12)
        # ax[i_s, i_t].set_y_label("# points", fontsize=12)
        ax[i_s, i_t].text(0.1, 24,
                          f'Median={np.median(_d2plot.precision):.2f} cm\nMean={np.mean(_d2plot.precision):.2f} cm\n'
                          f'80%={np.quantile(_d2plot.precision, 0.8):.2f} cm')
        ax[i_s, i_t].set_ylim([0.0, 30.0])
        ax[i_s, i_t].set_xlim([0, 0.2])

plt.savefig("precision.png", dpi=300)
