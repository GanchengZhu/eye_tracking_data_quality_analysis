import os

import pandas as pd

task_list = []
validation_time = []
rms_s2s = []
acc = []
subject_id = []

base_path = "../raw_data"
_subjects = os.listdir(base_path)

import numpy as np


def pixel_to_cm(raw_point, x_dpi=370.70248, y_dpi=372.31873):
    x_pixels_per_cm = x_dpi / 2.54
    y_pixels_per_cm = y_dpi / 2.54
    return np.array(raw_point) / np.array([x_pixels_per_cm, y_pixels_per_cm])


def d3(estimated_points, ground_truth_points, win_size=3, stride=1):
    """
    Calculate d3 from eye movement data.

    :param estimated_points: n * 2 array of estimated (x, y) points
    :param ground_truth_points: n * 2 array of ground truth (x, y) points
    :param win_size: window size, default is 3
    :param stride: step size for sliding window, default is 1
    :return: min values (precision, accuracy) over all windows
    """
    if len(estimated_points) != len(ground_truth_points):
        raise ValueError("estimated_points and ground_truth_points must have the same length")

    estimated_points = np.array(estimated_points)
    ground_truth_points = np.array(ground_truth_points)

    # Initialize the list to store d3 values
    d3_values = []
    acc_values = []
    std_values = []
    rms_s2s_values = []

    # Iterate through the points with the given window size and stride
    for i in range(0, len(estimated_points) - win_size + 1, stride):
        # Extract the current window of points
        est_window = estimated_points[i:i + win_size]
        gt_window = ground_truth_points[i:i + win_size]

        acc = np.sqrt((np.mean(gt_window[:, 0]) - np.mean(est_window[:, 0])) ** 2 +
                      (np.mean(gt_window[:, 1]) - np.mean(est_window[:, 1])) ** 2)

        std = np.sqrt(np.mean((est_window[:, 0] - np.mean(est_window[:, 0])) ** 2
                              + (est_window[:, 1] - np.mean(est_window[:, 1])) ** 2))

        acc_values.append(acc)
        std_values.append(std)
        d3_values.append(acc ** 2 * std ** 2)
        rms_s2s = np.sqrt(
            np.mean(np.diff(est_window[:, 0]) ** 2 + np.diff(est_window[:, 1]) ** 2))
        rms_s2s_values.append(rms_s2s)

    d3_arg_min = np.argmin(d3_values)
    # Return the average d3 value over all windows
    return acc_values[d3_arg_min], rms_s2s_values[d3_arg_min]


def data_analysis(data, task_id, subj_id, validation_index):
    rms_values = []
    acc_values = []
    gt_x_list = []
    gt_y_list = []
    x_gt, y_gt, x_estimated, y_estimated = data.x_gt.values, data.y_gt.values, \
        data.x_estimated.values, data.y_estimated.values

    start_index = 0
    cursor = 1
    while cursor < len(x_gt):
        if x_gt[cursor] == x_gt[cursor - 1] and y_gt[cursor] == y_gt[cursor - 1]:
            cursor += 1
            continue
        elif x_gt[cursor] != x_gt[cursor - 1] or y_gt[cursor] != y_gt[cursor - 1]:
            end_index = cursor - 1

            gt_points = list(zip(x_gt[start_index:end_index], y_gt[start_index:end_index]))
            estimated_points = list(zip(x_estimated[start_index:end_index], y_estimated[start_index:end_index]))

            acc, rms_s2s = d3(pixel_to_cm(estimated_points), pixel_to_cm(gt_points),  win_size=8, stride=1)
            acc_values.append(acc)
            rms_values.append(rms_s2s)
            tmp_point = pixel_to_cm(gt_points[0])
            gt_x_list.append(tmp_point[0])
            gt_y_list.append(tmp_point[1])
            start_index = cursor
            cursor += 1
            continue

    end_index = len(x_gt) - 1
    gt_points = list(zip(x_gt[start_index:end_index], y_gt[start_index:end_index]))
    estimated_points = list(zip(x_estimated[start_index:end_index], y_estimated[start_index:end_index]))

    acc, rms_s2s = d3(pixel_to_cm(estimated_points), pixel_to_cm(gt_points), win_size=8, stride=1)
    acc_values.append(acc)
    rms_values.append(rms_s2s)
    tmp_point = pixel_to_cm(gt_points[0])
    gt_x_list.append(tmp_point[0])
    gt_y_list.append(tmp_point[1])

    task_dir = f'./{task_id}_{"pre" if validation_index==0 else "post"}_task_evaluation'
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)

    pd.DataFrame({
        "accuracy": acc_values,
        "precision": rms_values,
        "gt_x": gt_x_list,
        "gt_y": gt_y_list,
    }).to_csv(os.path.join(task_dir, f"{subj_id}.csv"), index=False)
    return np.mean(acc_values), np.mean(rms_values)


def pointRMS_S2S(x_estimated, y_estimated):
    x_dpi = 370.70248
    y_dpi = 372.31873
    x_pixels_per_cm = x_dpi / 2.54
    y_pixels_per_cm = y_dpi / 2.54
    x_estimated = x_estimated / x_pixels_per_cm
    y_estimated = y_estimated / y_pixels_per_cm
    x_delta = np.diff(x_estimated)
    y_delta = np.diff(y_estimated)
    # print(x_delta)
    print(y_delta)
    return np.sqrt(np.sum(x_delta ** 2 + y_delta ** 2) / len(x_delta))


for _subj in _subjects:
    if _subj == ".DS_Store":
        continue
    print(_subj)

    for _folder in os.listdir(os.path.join(base_path, _subj)):
        print(_folder)
        if 'validation' in _folder:
            _task = _folder.split('_')[2]  # get task name
            _val_files = os.listdir(os.path.join(base_path, _subj, _folder))
            if len(_val_files) < 2:
                pass
            else:
                _v0_t = float('.'.join(_val_files[0].split('_')[-3:-1]))
                _v1_t = float('.'.join(_val_files[1].split('_')[-3:-1]))

                if _v1_t > _v0_t:
                    first_validation = os.path.join(base_path, _subj, _folder, _val_files[0])
                    last_validation = os.path.join(base_path, _subj, _folder, _val_files[1])
                else:
                    first_validation = os.path.join(base_path, _subj, _folder, _val_files[1])
                    last_validation = os.path.join(base_path, _subj, _folder, _val_files[0])

                first_vali_data = pd.read_csv(first_validation)
                last_vali_data = pd.read_csv(last_validation)
                task_list.append(_task)
                validation_time.append(0)
                subject_id.append(_subj)
                results = data_analysis(first_vali_data, _task, validation_index=0, subj_id=_subj)
                acc.append(results[0])
                rms_s2s.append(results[1])
                task_list.append(_task)
                validation_time.append(1)
                results = data_analysis(last_vali_data, _task, validation_index=1, subj_id=_subj)
                acc.append(results[0])
                rms_s2s.append(results[1])
                subject_id.append(_subj)

if not os.path.exists("./summary"):
    os.mkdir("./summary")
# break
pd.DataFrame({
    "sub_id": subject_id,
    "task": task_list,
    "validation": validation_time,
    "accuracy": acc,
    "precision": rms_s2s
}).to_csv("./summary/validation_summary.csv", index=False)
