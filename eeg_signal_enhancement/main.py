"""
Cross-Modal EEG Signal Enhancement (Case Study 1)

Uses concurrent multi-site IMU signals to estimate and subtract motion-induced
artifacts from walking EEG. Per-window two-phase model: Ridge init → Adam
fine-tune with combined Temporal + Frequency Loss.

Usage:
    python main.py

Outputs (written to results/):
    <task>_timeseries.png  — waveform comparison (Noisy / Cleaned / Baseline)
    <task>_denoised.csv    — denoised signal (columns: noisy, cleaned)
"""

import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from config import WALKING_FILES, RESULTS_DIR, TRAIN_RATIO
from core.data_loader import load_data
from core.losses import compute_ref_psd
from core.model import run_pipeline
from core.evaluate import post_process, eval_losses, plot_results

os.makedirs(RESULTS_DIR, exist_ok=True)


def process_task(walk_file):
    task = walk_file.replace(".csv", "")
    print(f"\n{'='*60}\n  {task}\n{'='*60}")

    eeg_noisy, imu_all, eeg_ref, eeg_base = load_data(walk_file)
    N       = len(eeg_noisy)
    N_train = int(N * TRAIN_RATIO)

    ref_rms = float(np.sqrt(np.mean(eeg_ref**2)))
    ref_psd = compute_ref_psd(eeg_ref)

    print("  Running pipeline ...")
    eeg_full    = run_pipeline(eeg_noisy, imu_all, eeg_ref, N_train, ref_rms, ref_psd)
    eeg_cleaned = post_process(eeg_full[N_train:])
    eeg_noisy_eval = eeg_noisy[N_train:]

    losses = eval_losses(eeg_cleaned, eeg_base)
    print(f"  T-loss={losses['temporal_loss']:.4f}  F-loss={losses['frequency_loss']:.4f}  "
          f"cleaned RMS={losses['rms_cleaned']:.1f}  baseline RMS={losses['rms_baseline']:.1f}")

    csv_path = os.path.join(RESULTS_DIR, f"{task}_denoised.csv")
    pd.DataFrame({"noisy": eeg_noisy_eval[:len(eeg_cleaned)], "cleaned": eeg_cleaned}).to_csv(csv_path, index=False)

    plot_results(
        eeg_noisy_eval, eeg_cleaned, eeg_base,
        title=task,
        save_path=os.path.join(RESULTS_DIR, f"{task}_timeseries.png"),
    )
    return {"task": task, **losses}


if __name__ == "__main__":
    all_losses = [process_task(f) for f in WALKING_FILES]

    print("\n" + "="*60)
    print(f"{'Task':<25} {'T-loss':>8} {'F-loss':>10} {'Cleaned RMS':>12}")
    print("-"*57)
    for r in all_losses:
        print(f"  {r['task']:<23} {r['temporal_loss']:>8.4f} {r['frequency_loss']:>10.4f} {r['rms_cleaned']:>10.1f} uV")
    print("Done.")
