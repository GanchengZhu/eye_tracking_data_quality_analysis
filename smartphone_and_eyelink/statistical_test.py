import numpy as np
import pandas as pd
from scipy.stats import stats

eyelink_path = "data_quality_results/eyelink/summary_eyelink.csv"
phone_path = "data_quality_results/phone/summary_phone.csv"

eyelink_df = pd.read_csv(eyelink_path)
phone_df = pd.read_csv(phone_path)

eyelink_acc = eyelink_df.Accuracy
eyelink_prec = eyelink_df.Precision
eyelink_me = eyelink_df.ME

phone_acc = phone_df.Accuracy
phone_prec = phone_df.Precision
phone_me = phone_df.ME

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

# Paired-sample t-test for ME
t_stat_me, p_value_me = stats.ttest_rel(phone_me, eyelink_me)
print(f"Paired-sample t-test for ME: t = {t_stat_me}, p = {p_value_me}")

# Calculate Cohen's d for ME (paired samples)
mean_diff_me = np.mean(phone_me - eyelink_me)
std_diff_me = np.std(phone_me - eyelink_me, ddof=1)
print(mean_diff_me, " ", std_diff_me)
cohens_d_me = mean_diff_me / std_diff_me
print(f"Cohen's d for ME: {cohens_d_me}")
