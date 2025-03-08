import glob
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from util import *

OUTPUT_DIR = "data_quality_results"
Path(f"{OUTPUT_DIR}/phone/subjects").mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_DIR}/eyelink/subjects").mkdir(parents=True, exist_ok=True)


def convert_cm_to_degrees(value_cm, distance_cm):
    return 2 * np.degrees(np.arctan(value_cm / (2 * distance_cm)))


def load_subject_data(subject_id):
    eyelink_path = glob.glob(f"dataset/eyelink/{subject_id:02d}/*.csv")
    phone_path = glob.glob(f"dataset/phone/{subject_id:02d}/*.txt")
    assert len(eyelink_path) == 1 and len(phone_path) == 1, f"Subject {subject_id:02d} data not found"
    eyelink_df = pd.read_csv(eyelink_path[0])
    phone_df = pd.read_csv(phone_path[0], delimiter=",")
    return eyelink_df, phone_df


def process_phone_position(position_df, window_size_ms, sample_rate):
    window_size = int(round(window_size_ms * sample_rate / 1000))
    if len(position_df) < window_size:
        return (np.nan,) * 6
    # for 24 points
    gaze_x = position_df.filteredX.values
    gaze_y = position_df.filteredY.values
    gt_x = position_df.gtX.values
    gt_y = position_df.gtY.values
    # distances from eyes to the phone screen
    distance = (position_df.leftDistance + position_df.rightDistance).values / 2
    metrics = []
    for i in range(len(gaze_x) - window_size + 1):
        gx, gy = gaze_x[i:i + window_size], gaze_y[i:i + window_size]
        tx, ty = gt_x[i:i + window_size].mean(), gt_y[i:i + window_size].mean()
        dist = distance[i:i + window_size]
        gx_deg = convert_cm_to_degrees(gx, dist)
        gy_deg = convert_cm_to_degrees(gy, dist)
        tx_deg = convert_cm_to_degrees(tx, dist)
        ty_deg = convert_cm_to_degrees(ty, dist)
        acc = np.sqrt((np.mean(gx_deg) - np.mean(tx_deg)) ** 2 + (np.mean(gy_deg) - np.mean(ty_deg)) ** 2)
        dx_deg, dy_deg = np.diff(gx_deg), np.diff(gy_deg)
        pre_rms_s2s = np.sqrt(np.mean(dx_deg ** 2 + dy_deg ** 2))
        _std = np.sqrt(np.mean((gx_deg - np.mean(gx_deg)) ** 2 + (gy_deg - np.mean(gy_deg)) ** 2))
        d3 = (acc ** 2) * (_std ** 2)
        # me in centi-meters
        me = np.mean(np.sqrt((gx - tx) ** 2 + (gy - ty) ** 2))
        metrics.append((acc, pre_rms_s2s, me, d3, tx, ty, dist.mean()))
    if not metrics:
        return (np.nan,) * 6
    best = min(metrics, key=lambda x: x[3] if not np.isnan(x[3]) else float('inf'))
    return best[0], best[1], best[2], best[4], best[5], best[6]


def process_eyelink_position(eyelink_df, phone_df, window_size_ms, sample_rate):
    window_size = int(round(window_size_ms * sample_rate / 1000))
    if len(eyelink_df) < window_size:
        return (np.nan,) * 6
    gaze_x = eyelink_df.right_x * EYELINK_SCALE_FACTOR_X_CM
    gaze_y = eyelink_df.right_y * EYELINK_SCALE_FACTOR_Y_CM
    metrics = []
    global_tx = phone_df.gtX.iloc[0]
    global_ty = phone_df.gtY.iloc[0]
    for i in range(len(gaze_x) - window_size + 1):
        start = eyelink_df.phone_timestamp.iloc[i]
        end = eyelink_df.phone_timestamp.iloc[i + window_size - 1]
        phone_window = phone_df[(phone_df.timestamp > start) & (phone_df.timestamp < end)]
        if phone_window.empty:
            continue
        # print(phone_df.leftDistance.values)
        dist = np.mean((phone_df.leftDistance + phone_df.rightDistance) / 2)
        # print(dist)
        # unit: cm
        tx, ty = phone_window.gtX.mean(), phone_window.gtY.mean()
        gx, gy = gaze_x.iloc[i:i + window_size].values, gaze_y.iloc[i:i + window_size].values

        # convert cm to degrees
        gx_deg = convert_cm_to_degrees(gx, dist)
        gy_deg = convert_cm_to_degrees(gy, dist)
        tx_deg = convert_cm_to_degrees(tx, dist)
        ty_deg = convert_cm_to_degrees(ty, dist)
        acc = np.sqrt((np.mean(gx_deg) - tx_deg) ** 2 + (np.mean(gy_deg) - ty_deg) ** 2)
        dx_deg, dy_deg = np.diff(gx_deg), np.diff(gy_deg)
        pre_rms_s2s = np.sqrt(np.mean(dx_deg ** 2 + dy_deg ** 2))
        _std = np.sqrt(np.mean((gx_deg - np.mean(gx_deg)) ** 2 + (gy_deg - np.mean(gy_deg)) ** 2))
        d3 = (acc ** 2) * (_std ** 2)

        if np.sum(gx < -200 / 1080 * 7.400):
            continue

        if np.sum(gx > 1280 / 1080 * 7.400):
            continue

        if np.sum(gy < -200 / 2249 * 15.34293):
            continue

        if np.sum(gy > 2449 / 2249 * 15.34293):
            continue

        if d3 == 0:
            continue

        if acc > 5:
            continue

        # me in centi-meters
        me = np.mean(np.sqrt((gx - tx) ** 2 + (gy - ty) ** 2))
        metrics.append((acc, pre_rms_s2s, me, d3, dist))
    if not metrics:
        res = [np.nan] * 6
        res[3] = global_tx
        res[4] = global_ty
        return res
    best = min(metrics, key=lambda x: x[3] if not np.isnan(x[3]) else float('inf'))
    return best[0], best[1], best[2], global_tx, global_ty, best[4]


def save_metrics(device, subject_id, metrics):
    df = pd.DataFrame(metrics, columns=["accuracy", "precision", "me", "gt_x", "gt_y", "distance"])
    df.to_csv(f"{OUTPUT_DIR}/{device}/subjects/{device}_{subject_id:02d}.csv", index=False)


def main():
    phone_summary, eyelink_summary = [], []
    for subject_id in tqdm(range(1, 33)):
        if subject_id in EYELINK_SKIP_IDS:
            continue

        eyelink_df, phone_df = load_subject_data(subject_id)
        stationary_phone_cm = phone_df[phone_df.showGaze == 1].copy()
        # convert pixels to centimeters
        stationary_phone_cm['filteredX'] *= PHONE_SCREEN_WIDTH_CM / PHONE_SCREEN_WIDTH_PIXELS
        stationary_phone_cm['gtX'] *= PHONE_SCREEN_WIDTH_CM / PHONE_SCREEN_WIDTH_PIXELS
        stationary_phone_cm['filteredY'] *= PHONE_SCREEN_HEIGHT_CM / PHONE_SCREEN_HEIGHT_PIXELS
        stationary_phone_cm['gtY'] *= PHONE_SCREEN_HEIGHT_CM / PHONE_SCREEN_HEIGHT_PIXELS

        # calculate phone sampling rate
        phone_times = phone_df.timestamp.values
        # get duration from nanoseconds to seconds
        duration = (phone_times[-1] - phone_times[0]) / 1e9
        phone_rate = len(phone_times) / duration if duration > 0 else 0
        phone_data_retention = phone_df.trackingState.mean()
        phone_metrics, eyelink_metrics = [], []
        for pos in range(24):
            pos_phone = stationary_phone_cm[stationary_phone_cm.positionID == pos]
            if pos_phone.empty:
                phone_metrics.append((np.nan,) * 6)
                eyelink_metrics.append((np.nan,) * 6)
                continue
            start, end = pos_phone.timestamp.iloc[0], pos_phone.timestamp.iloc[-1]
            pos_eyelink = eyelink_df[(eyelink_df.phone_timestamp > start) & (eyelink_df.phone_timestamp < end)]
            pm = process_phone_position(pos_phone, WINDOW_SIZE_MS, phone_rate)
            em = process_eyelink_position(pos_eyelink, pos_phone, WINDOW_SIZE_MS,
                                          EYELINK_SAMPLE_RATE) if not pos_eyelink.empty else (np.nan,) * 6
            phone_metrics.append(pm)
            eyelink_metrics.append(em)
        save_metrics("phone", subject_id, phone_metrics)
        save_metrics("eyelink", subject_id, eyelink_metrics)
        phone_acc = np.nanmean([m[0] for m in phone_metrics])
        phone_pre_rms_s2s = np.nanmean([m[1] for m in phone_metrics])
        phone_me = np.nanmean([m[2] for m in phone_metrics])
        eyelink_acc = np.nanmean([m[0] for m in eyelink_metrics])
        eyelink_pre_rms_s2s = np.nanmean([m[1] for m in eyelink_metrics])
        eyelink_me = np.nanmean([m[2] for m in eyelink_metrics])
        phone_summary.append([subject_id, phone_acc, phone_pre_rms_s2s, phone_me, 1 - phone_data_retention, phone_rate])
        eyelink_summary.append([subject_id, eyelink_acc, eyelink_pre_rms_s2s, eyelink_me])
    pd.DataFrame(phone_summary, columns=["ID", "Accuracy", "Precision", "ME", "DataLoss", "SampleRate"]).to_csv(
        f"{OUTPUT_DIR}/phone/summary_phone.csv", index=False)
    pd.DataFrame(eyelink_summary, columns=["ID", "Accuracy", "Precision", "ME"]).to_csv(
        f"{OUTPUT_DIR}/eyelink/summary_eyelink.csv", index=False)


if __name__ == "__main__":
    main()
