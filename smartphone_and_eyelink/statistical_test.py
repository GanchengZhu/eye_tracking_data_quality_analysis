import numpy as np
import pandas as pd
from scipy.stats import stats

eyelink_path = "results_correction/eyelink/summary_eyelink.csv"
phone_path = "results_correction/phone/summary_phone.csv"

eyelink_df = pd.read_csv(eyelink_path)
phone_df = pd.read_csv(phone_path)

eyelink_acc = eyelink_df.Accuracy
eyelink_prec = eyelink_df.Precision

phone_acc = phone_df.Accuracy
phone_prec = phone_df.Precision

# Paired-sample t-test for Accuracy
t_stat_accuracy, p_value_accuracy = stats.ttest_rel(phone_acc, eyelink_acc)
print(f"Paired-sample t-test for Accuracy: t = {t_stat_accuracy}, p = {p_value_accuracy}")

# Calculate Cohen's d for Accuracy (paired samples)
mean_diff_accuracy = np.mean(phone_acc - eyelink_acc)
std_diff_accuracy = np.std(phone_acc - eyelink_acc, ddof=1)
cohens_d_accuracy = mean_diff_accuracy / std_diff_accuracy
print(f"Cohen's d for Accuracy: {cohens_d_accuracy}")

# Paired-sample t-test for Precision
t_stat_precision, p_value_precision = stats.ttest_rel(phone_prec, eyelink_prec)
print(f"Paired-sample t-test for Precision: t = {t_stat_precision}, p = {p_value_precision}")

# Calculate Cohen's d for Precision (paired samples)
mean_diff_precision = np.mean(phone_prec - eyelink_prec)
std_diff_precision = np.std(phone_prec - eyelink_prec, ddof=1)
print(mean_diff_precision, " ", std_diff_precision)
cohens_d_precision = mean_diff_precision / std_diff_precision
print(f"Cohen's d for Precision: {cohens_d_precision}")

