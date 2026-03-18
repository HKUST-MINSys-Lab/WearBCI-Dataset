import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from config import (FS_EEG, WIN_SEC, HOP_SEC, LAG_MS, LAG_STEP,
                    RIDGE_ALPHA, FINETUNE_STEPS, LR, LAMBDA_T, LAMBDA_F, LAMBDA_R)
from core.losses import temporal_loss, frequency_loss


def lag_features(imu_window, fs=FS_EEG):
    """Construct lagged IMU feature matrix for a single window."""
    lag_max = int(LAG_MS * fs / 1000)
    step    = max(1, int(LAG_STEP * fs / 1000))
    lags    = range(-lag_max, lag_max + 1, step)
    return np.hstack([np.roll(imu_window, k, axis=0) for k in lags])


def train_window(X_w, art_w, noisy_w, ref_rms, ref_psd, scaler):
    """
    Two-phase per-window training:
      Phase 1 — Ridge regression for closed-form initialisation.
      Phase 2 — Adam fine-tuning with combined temporal + frequency loss.
    Returns (weights, bias) as numpy arrays.
    """
    X_s = scaler.transform(X_w).astype(np.float32)

    # Phase 1: Ridge closed-form initialisation
    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    ridge.fit(X_s, art_w)
    w0, b0 = ridge.coef_.astype(np.float32), float(ridge.intercept_)

    # Phase 2: Adam fine-tuning
    X_t     = torch.tensor(X_s, dtype=torch.float32)
    noisy_t = torch.tensor(noisy_w.astype(np.float32), dtype=torch.float32)
    model   = nn.Linear(X_s.shape[1], 1, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(w0).unsqueeze(0))
        model.bias.fill_(b0)

    optim = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(FINETUNE_STEPS):
        optim.zero_grad()
        cleaned = noisy_t - model(X_t).squeeze()
        loss = (LAMBDA_T * temporal_loss(cleaned, ref_rms)
                + LAMBDA_F * frequency_loss(cleaned, ref_psd)
                + LAMBDA_R * torch.sum(model.weight**2))
        loss.backward()
        optim.step()

    return model.weight.detach().numpy().flatten(), float(model.bias.detach())


def run_pipeline(eeg_noisy, imu_all, eeg_ref, N_train, ref_rms, ref_psd, fs=FS_EEG):
    """
    Slide a Hann-windowed frame over the full signal, train a model per window,
    and reconstruct the cleaned signal via overlap-add.
    """
    N    = len(eeg_noisy)
    win  = int(WIN_SEC * fs)
    hop  = int(HOP_SEC * fs)
    hann = np.hanning(win)
    artifact = eeg_noisy - eeg_ref

    scaler = StandardScaler().fit(np.vstack([
        lag_features(imu_all[s: s + win])
        for s in range(0, N_train - win + 1, hop)
    ]))

    out_sum = np.zeros(N)
    wgt_sum = np.zeros(N)
    n_train = n_eval = 0

    for start in range(0, N - win + 1, hop):
        end = start + win
        X_w = lag_features(imu_all[start:end])
        w, b = train_window(X_w, artifact[start:end], eeg_noisy[start:end],
                            ref_rms, ref_psd, scaler)
        X_ws      = scaler.transform(X_w).astype(np.float32)
        cleaned_w = eeg_noisy[start:end] - (X_ws @ w + b)
        out_sum[start:end] += cleaned_w * hann
        wgt_sum[start:end] += hann
        n_train += start < N_train
        n_eval  += start >= N_train

    print(f"  train windows={n_train}  eval windows={n_eval}")
    mask = wgt_sum > 1e-10
    return np.where(mask, out_sum / wgt_sum, eeg_noisy)
