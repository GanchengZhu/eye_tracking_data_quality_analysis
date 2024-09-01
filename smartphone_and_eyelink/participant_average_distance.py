import glob

import numpy as np
import pandas as pd

file_lists = glob.glob("results_correction/phone/subjects/phone_*.csv")

distance_list = []
for f in file_lists:
    df = pd.read_csv(f)
    avg_distance = df.distance
    distance_list.append(avg_distance)

print(np.mean(distance_list))
print(np.std(distance_list))