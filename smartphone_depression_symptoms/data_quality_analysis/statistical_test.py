import pandas as pd
from scipy.stats import ttest_rel

# 读取CSV文件
cal_res = pd.read_csv('./summary/validation_summary.csv', engine='python')

# Initialize lists to store results
results = []


# 按task分组并分别处理每个task
for task, task_data in cal_res.groupby('task'):
    # 确保每个sub_id和validation的组合只有一个值
    task_data = task_data.groupby(['sub_id', 'validation']).agg({'accuracy': 'mean', 'precision': 'mean'}).reset_index()

    # 将数据透视表以便于成对样本T检验
    paired_data = task_data.pivot(index='sub_id', columns='validation', values=['accuracy', 'precision'])

    # 分别提取出第一次和第二次验证的accuracy和precision
    accuracy_0 = paired_data['accuracy'][0]
    accuracy_1 = paired_data['accuracy'][1]
    precision_0 = paired_data['precision'][0]
    precision_1 = paired_data['precision'][1]

    # 进行成对样本T检验
    t_stat_accuracy, p_value_accuracy = ttest_rel(accuracy_0, accuracy_1)
    t_stat_precision, p_value_precision = ttest_rel(precision_0, precision_1)

    # Calculate means and standard deviations
    accuracy_0_mean, accuracy_0_std = accuracy_0.mean(), accuracy_0.std()
    accuracy_1_mean, accuracy_1_std = accuracy_1.mean(), accuracy_1.std()
    precision_0_mean, precision_0_std = precision_0.mean(), precision_0.std()
    precision_1_mean, precision_1_std = precision_1.mean(), precision_1.std()

    # Store results
    results.append({
        'Task': task,
        'Validation Type': 'First validation',
        'Accuracy': f"{accuracy_0_mean:.3f}±{accuracy_0_std:.3f}",
        'Precision': f"{precision_0_mean:.3f}±{precision_0_std:.3f}"
    })
    results.append({
        'Task': task,
        'Validation Type': 'Last validation',
        'Accuracy': f"{accuracy_1_mean:.3f}±{accuracy_1_std:.3f}",
        'Precision': f"{precision_1_mean:.3f}±{precision_1_std:.3f}"
    })
    results.append({
        'Task': task,
        'Validation Type': 'First vs Last',
        'Accuracy': f"t-stat = {t_stat_accuracy:.2f}, p = {p_value_accuracy:.4f}",
        'Precision': f"t-stat = {t_stat_precision:.2f}, p = {p_value_precision:.4f}"
    })


# Convert results to DataFrame for better presentation
results_df = pd.DataFrame(results)
results_df.to_csv('./summary/validation_results.csv', index=False)
print(results_df)