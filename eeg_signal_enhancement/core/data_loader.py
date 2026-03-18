import os
import numpy as np
import pandas as pd
from scipy.signal import firwin, filtfilt

from config import EEG_DIR, IMU_DIR, IMU_PARTS, CHANNEL, FS_EEG, BP_LOW, BP_HIGH


def bandpass(data, lo=BP_LOW, hi=BP_HIGH, fs=FS_EEG):
    ntaps = int(fs // 2) | 1
    coef  = firwin(ntaps, [lo, hi], pass_zero=False, fs=fs)
    out   = filtfilt(coef, 1.0, data - np.mean(data))
    return out - np.mean(out)


def load_eeg(filename):
    df = pd.read_csv(os.path.join(EEG_DIR, filename))
    return bandpass(df[CHANNEL].values.astype(float))


def load_imu(part, filename, eeg_len, fs=FS_EEG):
    df = pd.read_csv(os.path.join(IMU_DIR, part, filename))
    df.columns = df.columns.str.strip()
    times = pd.to_datetime(df["时间"]).values.astype(np.int64) / 1e9
    times -= times[0]
    t_eeg     = np.arange(eeg_len) / fs
    acc_cols  = [c for c in df.columns if "加速度" in c][:3]
    gyro_cols = [c for c in df.columns if "角速度" in c][:3]
    raw       = df[acc_cols + gyro_cols].values.astype(float)
    imu_up    = np.column_stack([np.interp(t_eeg, times, raw[:, i]) for i in range(raw.shape[1])])
    return np.column_stack([bandpass(imu_up[:, i]) for i in range(6)])


def load_data(walk_file):
    """Load noisy EEG, all IMU channels, static reference EEG, and quiet baseline EEG."""
    eeg_noisy = load_eeg(walk_file)
    N = len(eeg_noisy)

    imu_list = []
    for part in IMU_PARTS:
        try:
            imu_list.append(load_imu(part, walk_file, N))
        except Exception as e:
            print(f"  [warn] IMU {part}: {e}")
    imu_all = np.hstack(imu_list) if imu_list else np.zeros((N, 1))

    eeg_ref  = load_eeg("walking_static_0.csv")
    eeg_ref  = np.tile(eeg_ref, N // len(eeg_ref) + 1)[:N]
    eeg_base = load_eeg("baseline_quiet.csv")

    rms = lambda x: float(np.sqrt(np.mean(x**2)))
    print(f"  noisy RMS={rms(eeg_noisy):.1f}  ref RMS={rms(eeg_ref):.1f}  "
          f"baseline RMS={rms(eeg_base):.1f}  IMU={imu_all.shape}")
    return eeg_noisy, imu_all, eeg_ref, eeg_base
