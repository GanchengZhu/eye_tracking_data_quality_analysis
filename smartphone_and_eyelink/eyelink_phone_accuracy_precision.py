import glob
import os.path

import numpy as np
import pandas as pd


def covertX(x):
    x /= 1080
    x *= 7.4000
    return x


def covertY(y):
    y /= 2249
    y *= 15.34293
    return y


def to_degree_x(x, distance):
    degree = 2 * np.rad2deg(np.arctan(x / 2 / distance))
    return degree


def to_degree_y(y, distance):
    degree = 2 * np.rad2deg(np.arctan(y / 2 / distance))
    return degree


EYELINK_SAMPLE_RATE = 1000
eyelink_accuracy_with_nan = []
phone_accuracy = []
eyelink_accuracy = []

phone_precision = []
eyelink_precision = []
eyelink_precision_with_nan = []

eyelink_skip_id = [6, 23, 31, 32]

eyelink_data_loss = []
phone_data_loss = []
phone_sample_rate = []

id_list = []

# the scale factor for converting Eyelink output capture video dimensions to phone dimensions
SCALE_FACTOR = (1080 / 332, 2249 / 692)

for subject_id in range(1, 33):
    # eyelink data path
    eyelink_data_path = glob.glob("dataset/eyelink/%02d/*.csv" % subject_id)
    # print(eyelink_data_path)
    assert len(eyelink_data_path) == 1
    eyelink_data_path = eyelink_data_path[0]

    # phone data path
    phone_data_path = glob.glob("dataset/phone/%02d/*.txt" % subject_id)
    assert len(phone_data_path) == 1
    phone_data_path = phone_data_path[0]
    # print(phone_data_path)
    phone_data = pd.read_csv(phone_data_path, delimiter=",")

    # read csv file using pandas
    phone_data_df = pd.read_csv(phone_data_path)
    eyelink_data_df = pd.read_csv(eyelink_data_path)

    # # phone timestamp in millisecond
    # phone_data_timestamp = phone_data_df.timestamp - phone_data_df.timestamp[0]
    # phone_data_timestamp = phone_data_timestamp / 1E6
    # # print(phone_t[len(phone_t) - 1])

    stationary_target_phone_data_df = phone_data_df[phone_data_df['showGaze'] == 1]
    # convert pixel into centimeter
    stationary_target_phone_data_df.loc[:, "filteredX"] = (
        stationary_target_phone_data_df.filteredX.apply(covertX))
    stationary_target_phone_data_df.loc[:, "gtX"] = (
        stationary_target_phone_data_df.gtX.apply(covertX))

    stationary_target_phone_data_df.loc[:, "filteredY"] = stationary_target_phone_data_df.filteredY.apply(covertY)
    stationary_target_phone_data_df.loc[:, "gtY"] = stationary_target_phone_data_df.gtY.apply(covertY)

    # eyelink data loss
    # available data / all data
    n_all_data = 0
    n_loss_data = 0
    for position_id in range(24):
        target_phone_data_df = stationary_target_phone_data_df[
            stationary_target_phone_data_df.positionID == position_id]

        # print(target_phone_data_df.timestamp[0])
        # screen data
        start_timestamp = target_phone_data_df.timestamp.tolist()[0]
        end_timestamp = target_phone_data_df.timestamp.tolist()[len(target_phone_data_df.timestamp) - 1]
        eyelink_index = (eyelink_data_df.phone_timestamp > start_timestamp) & (
                eyelink_data_df.phone_timestamp < end_timestamp)
        target_eyelink_data_df = eyelink_data_df[eyelink_index]
        # print(f'target_eyelink_data_df length: {len(target_eyelink_data_df)}')

        left_loss = (target_eyelink_data_df.left_x == -32768.0) & (target_eyelink_data_df.left_y == -32768.0)
        right_loss = (target_eyelink_data_df.left_x == -32768.0) & (target_eyelink_data_df.left_y == -32768.0)
        n_loss_data += np.sum(left_loss & right_loss)
        n_all_data += len(target_eyelink_data_df)
    if (n_loss_data / n_all_data) > 0.15:
        print(
            f'subject id: {subject_id}, n_all_data: {n_all_data},'
            f' n_loss_data: {n_loss_data}, loss: {n_loss_data / n_all_data}')
        continue

    phone_data_loss.append(np.sum(phone_data.trackingState) / len(phone_data))
    # convert nanotime timestamp into millisecond
    phone_t = phone_data.timestamp - phone_data.timestamp[0]
    phone_t = phone_t * 1E-6  # convert to millisecond
    # Calculate sample rate from calibration start to validation ending
    _sample_rate = len(phone_t) * 1E3 / (phone_t[len(phone_t) - 1] - phone_t[0])
    phone_sample_rate.append(
        _sample_rate
    )

    # error
    subject_phone_accuracy = []
    subject_phone_precision = []
    subject_phone_target_position_x = []
    subject_phone_target_position_y = []
    subject_phone_distance = []

    subject_eyelink_accuracy = []
    subject_eyelink_precision = []
    subject_eyelink_target_position_x = []
    subject_eyelink_target_position_y = []
    subject_eyelink_distance = []

    for position_id in range(24):
        # analysis phone
        target_phone_data_df = stationary_target_phone_data_df[
            stationary_target_phone_data_df.positionID == position_id]
        # subject_phone_target_position_x.append(target_phone_data_df.gtX[0])
        # subject_phone_target_position_y.append(target_phone_data_df.gtY[0])
        # print(target_phone_data_df.timestamp[0])
        # screen data
        start_timestamp = target_phone_data_df.timestamp.tolist()[0]
        end_timestamp = target_phone_data_df.timestamp.tolist()[len(target_phone_data_df.timestamp) - 1]
        eyelink_index = (eyelink_data_df.phone_timestamp > start_timestamp) & (
                eyelink_data_df.phone_timestamp < end_timestamp)
        target_eyelink_data_df = eyelink_data_df[eyelink_index]
        # print(f'target_eyelink_data_df length: {len(target_eyelink_data_df)}')

        # left__loss = (target_eyelink_data_df.left_x == -32768.0) & (target_eyelink_data_df.left_y == -32768.0)
        # right_loss = (target_eyelink_data_df.left_x == -32768.0) & (target_eyelink_data_df.left_y == -32768.0)

        # initialize temporary variable
        d3_phone_list = []
        acc_phone_list = []
        precision_phone_list = []
        distance_phone_list = []

        d3_eyelink_list = []
        acc_eyelink_list = []
        precision_eyelink_list = []
        distance_eyelink_list = []

        # calculate acc and precision with the sliding window.
        # window size = 8 (about 264 ~ 307 ms), stride = 1,
        phone_win_size = int(np.round(175 / (1000 / _sample_rate)))
        for k in range(len(target_phone_data_df) - phone_win_size + 1):
            eye_to_screen_distance = (target_phone_data_df.leftDistance[k:k + phone_win_size]
                                      + target_phone_data_df.rightDistance[k:k + phone_win_size]) / 2
            # eye_to_screen_distance = target_phone_data_df.leftDistance[k:k + phone_win_size]

            # x = target_phone_data_df.filteredX[k:k + phone_win_size] * 360 / (2 * np.pi * eye_to_screen_distance)
            # y = target_phone_data_df.filteredY[k:k + phone_win_size] * 360 / (2 * np.pi * eye_to_screen_distance)
            # gt_x = target_phone_data_df.gtX[k:k + phone_win_size] * 360 / (2 * np.pi * eye_to_screen_distance)
            # gt_y = target_phone_data_df.gtY[k:k + phone_win_size] * 360 / (2 * np.pi * eye_to_screen_distance)
            x = to_degree_x(target_phone_data_df.filteredX[k:k + phone_win_size], eye_to_screen_distance)
            y = to_degree_y(target_phone_data_df.filteredY[k:k + phone_win_size], eye_to_screen_distance)
            gt_x = to_degree_x(target_phone_data_df.gtX[k:k + phone_win_size], eye_to_screen_distance)
            gt_y = to_degree_y(target_phone_data_df.gtY[k:k + phone_win_size], eye_to_screen_distance)

            dva = np.sqrt((np.mean(gt_x) - np.mean(x)) ** 2 +
                          (np.mean(gt_y) - np.mean(y)) ** 2)
            # eye_to_screen_distance = (target_phone_data_df.leftDistance[k:k + phone_win_size]
            #                                   + target_phone_data_df.rightDistance[k:k + phone_win_size]) / 2
            distance_phone_list.append(eye_to_screen_distance)
            acc_phone_list.append(dva)

            std = np.sqrt(np.mean((x - np.mean(x)) ** 2 + (y - np.mean(y)) ** 2))
            d3 = (dva ** 2 * std ** 2)

            d3_phone_list.append(d3)

            rms_s2s = np.sqrt(np.mean(np.diff(x) ** 2 + np.diff(y) ** 2))
            precision_phone_list.append(rms_s2s)

        subject_phone_distance.append(np.mean(distance_phone_list))
        # print(d_list)
        # print(acc_phone_list)
        arg_min = np.argmin(d3_phone_list)

        precision = precision_phone_list[arg_min]
        accuracy = acc_phone_list[arg_min]
        subject_phone_accuracy.append(accuracy)
        subject_phone_precision.append(precision)
        # print(arg_min)
        gt_x = np.mean(target_phone_data_df.gtX[arg_min:arg_min + phone_win_size])
        gt_y = np.mean(target_phone_data_df.gtY[arg_min:arg_min + phone_win_size])
        subject_phone_target_position_x.append(gt_x)
        subject_phone_target_position_y.append(gt_y)
        # analysis eyelink
        # TO-DO
        eyelink_win_size = int(np.round(175 / (1000 / 500)))
        global_gt_x = None
        global_gt_y = None
        for k in range(len(target_eyelink_data_df) - eyelink_win_size + 1):
            target_window_eyelink_data_df = target_eyelink_data_df[k: k + eyelink_win_size]
            _timestamp = target_window_eyelink_data_df.phone_timestamp.tolist()
            eyelink_start_timestamp = _timestamp[0]
            eyelink_end_timestamp = _timestamp[-1]

            phone_index = ((target_phone_data_df.timestamp > start_timestamp) &
                           (target_phone_data_df.timestamp < end_timestamp))

            eye_to_screen_distance = (target_phone_data_df.leftDistance[phone_index]
                                      + target_phone_data_df.rightDistance[phone_index]) / 2


            x = covertX(target_window_eyelink_data_df.right_x * SCALE_FACTOR[0])
            y = covertY(target_window_eyelink_data_df.right_y * SCALE_FACTOR[1])

            # it has already converted from pixel to centimeter
            gt_x = target_phone_data_df.gtX[phone_index]
            gt_y = target_phone_data_df.gtY[phone_index]

            contains_nan = (True in np.isnan(x)) and (True in np.isnan(y))
            if contains_nan: continue

            mean_distance = np.mean(eye_to_screen_distance)

            mean_x = np.mean(x)
            mean_y = np.mean(y)
            mean_gt_x = np.mean(gt_x)
            mean_gt_y = np.mean(gt_y)

            global_gt_x = mean_gt_x
            global_gt_y = mean_gt_y

            gt_degree_x = to_degree_x(mean_gt_x, mean_distance)
            degree_x = to_degree_x(mean_x, mean_distance)

            gt_degree_y = to_degree_y(mean_gt_y, mean_distance)
            degree_y = to_degree_y(mean_y, mean_distance)
            dva = np.sqrt((gt_degree_x - degree_x) ** 2 +
                          (gt_degree_y - degree_y) ** 2)

            std = np.sqrt(np.mean((x - np.mean(x)) ** 2 + (y - np.mean(y)) ** 2))
            d3 = (dva ** 2 * std ** 2)

            if np.sum(x < -200 / 1080 * 7.400):
                continue

            if np.sum(x > 1280 / 1080 * 7.400):
                continue

            if np.sum(y < -200 / 2249 * 15.34293):
                continue

            if np.sum(y > 2449 / 2249 * 15.34293):
                continue

            if d3 == 0:
                continue

            if dva > 5:
                continue

            acc_eyelink_list.append(dva)
            d3_eyelink_list.append(d3)
            rms_s2s = np.sqrt(
                np.mean(np.diff(to_degree_x(x, mean_distance)) ** 2 + np.diff(to_degree_y(y, mean_distance)) ** 2))
            precision_eyelink_list.append(rms_s2s)
            distance_eyelink_list.append(mean_distance)
        if len(d3_eyelink_list) > 0:
            subject_eyelink_distance.append(np.mean(distance_phone_list))
            # print(d_list)
            # print(acc_phone_list)
            arg_min = np.argmin(d3_eyelink_list)

            precision = precision_eyelink_list[arg_min]
            accuracy = acc_eyelink_list[arg_min]
            # if accuracy < 10:
            subject_eyelink_accuracy.append(accuracy)
            subject_eyelink_precision.append(precision)
        else:
            subject_eyelink_distance.append(np.nan)
            subject_eyelink_accuracy.append(np.nan)
            subject_eyelink_precision.append(np.nan)
        # else:
        #     # filter outlier
        #     subject_eyelink_accuracy.append(np.nan)
        #     subject_eyelink_precision.append(np.nan)
        # print(arg_min)
        gt_x = np.mean(global_gt_x)
        gt_y = np.mean(global_gt_y)
        subject_eyelink_target_position_x.append(gt_x)
        subject_eyelink_target_position_y.append(gt_y)

        # left__loss = (target_eyelink_data_df.left_x == -32768.0) & (target_eyelink_data_df.left_y == -32768.0)
        # right_loss = (target_eyelink_data_df.left_x == -32768.0) & (target_eyelink_data_df.left_y == -32768.0)

    if not os.path.exists("results/phone/subjects"):
        os.makedirs("results/phone/subjects")

    if not os.path.exists("results/eyelink/subjects"):
        os.makedirs("results/eyelink/subjects")

    # save phone subject data
    pd.DataFrame(
        np.array([subject_phone_accuracy, subject_phone_precision, subject_phone_target_position_x,
                  subject_phone_target_position_y, subject_phone_distance]).T,
        columns=["accuracy", "precision", "gt_x", "gt_y", "distance"]
    ).to_csv(os.path.join("results/phone/subjects", "phone_%02d.csv" % subject_id), index=False)

    phone_accuracy.append(np.mean(subject_phone_accuracy))
    phone_precision.append(np.mean(subject_phone_precision))

    # save eyelink subject data
    pd.DataFrame(
        np.array([subject_eyelink_accuracy, subject_eyelink_precision, subject_eyelink_target_position_x,
                  subject_eyelink_target_position_y, subject_eyelink_distance]).T,
        columns=["accuracy", "precision", "gt_x", "gt_y", "distance"]
    ).to_csv(os.path.join("results/eyelink/subjects", "eyelink_%02d.csv" % subject_id), index=False)

    eyelink_accuracy.append(np.nanmean(subject_eyelink_accuracy))
    eyelink_precision.append(np.nanmean(subject_eyelink_precision))

    id_list.append(subject_id)

# print(len(id_list))
# print(len(phone_accuracy))
# print(np.array([id_list, phone_accuracy, phone_precision, phone_data_loss, phone_sample_rate]).shape)

pd.DataFrame(
    np.array([id_list, phone_accuracy, phone_precision, phone_data_loss, phone_sample_rate]).T,
    columns=["ID", "Accuracy", "Precision", "DataLoss", "SampleRate"]
).to_csv(os.path.join("results/phone", "summary_phone.csv"), index=False)

pd.DataFrame(
    np.array([id_list, eyelink_accuracy, eyelink_precision]).T,
    columns=["ID", "Accuracy", "Precision"]
).to_csv(os.path.join("results/eyelink", "summary_eyelink.csv"), index=False)
#     # target_timestamp.timestamp
#     error_list = []
#     for j in range(23):
#         point_phone_df = fix_phone_df[fix_phone_df.positionID == j]
#         min_error = 1E9
#         for k in range(len(point_phone_df) - 7):
#             x = np.mean(point_phone_df.filteredX[k:k + 8])
#             y = np.mean(point_phone_df.filteredY[k:k + 8])
#             gt_x = np.mean(point_phone_df.gtX[k:k + 8])
#             gt_y = np.mean(point_phone_df.gtY[k:k + 8])
#             error = (gt_x - x) ** 2 + (gt_y - y) ** 2
#             error = np.sqrt(error)
#             if error < min_error:
#                 min_error = error
#         if min_error != 1E9:
#             error_list.append(min_error)
#
#     print(np.mean(error_list))
#     result.append([i, np.mean(error_list), len(np.unique(fix_phone_df.positionID))])
#
#     # plt.scatter(x=phone_t, y=fix_phone_df.filteredX, color='k', label='Phone-ES-X', linewidths=1)
#     # plt.scatter(x=phone_t, y=fix_phone_df.gtX, color='r', label='Phone-GT-X', linewidths=1)
#     # # plt.scatter(x=eyelink_t, y=eyelink_x,  color='b', label='EyeLink-ES', linewidths=1)
#     # plt.legend()
#     # plt.savefig("figures/phone/" + i.split("\\")[-2] + "_x.png")
#     # plt.close()
#     #
#     # plt.scatter(x=phone_t, y=fix_phone_df.filteredY, color='b', label='Phone-ES-Y', linewidths=1)
#     # plt.scatter(x=phone_t, y=fix_phone_df.gtY, color='r', label='Phone-GT-Y', linewidths=1)
#     # # plt.scatter(x=eyelink_t, y=eyelink_y, color='k', label='EyeLink-ES', linewidths=1)
#     # plt.legend()
#     # plt.savefig("figures/phone/" + i.split("\\")[-2] + "_y.png")
#     # plt.close()
# print(result)
# # print(np.mean(result))
# # print(np.std(result))
# # print(np.median(result))
#
#
# pd.DataFrame(result).to_csv("result_preprocessing.csv", index=False)
