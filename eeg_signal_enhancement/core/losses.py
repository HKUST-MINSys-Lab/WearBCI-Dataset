import numpy as np
import torch

from config import FS_EEG, BP_LOW, BP_HIGH, NPERSEG


def compute_ref_psd(signal, fs=FS_EEG):
    """Compute reference PSD from a numpy signal, returned as a torch tensor."""
    n_segs  = max(1, len(signal) // NPERSEG)
    psd_sum = np.zeros(NPERSEG // 2 + 1, dtype=np.float32)
    valid   = 0
    for i in range(n_segs):
        seg = signal[i * NPERSEG: (i + 1) * NPERSEG]
        if len(seg) < NPERSEG:
            break
        fft_v    = np.fft.rfft(seg)
        psd_sum += (fft_v.real**2 + fft_v.imag**2) / NPERSEG
        valid   += 1
    return torch.tensor(psd_sum / max(valid, 1), dtype=torch.float32)


def temporal_loss(cleaned, ref_rms):
    """Penalise RMS deviation from the static reference."""
    rms = torch.sqrt(torch.mean(cleaned**2) + 1e-12)
    return torch.abs(rms - ref_rms) / (ref_rms + 1e-10)


def frequency_loss(cleaned, ref_psd, fs=FS_EEG):
    """Penalise PSD deviation from the static reference within the passband."""
    N      = cleaned.shape[0]
    freqs  = torch.fft.rfftfreq(NPERSEG, d=1.0 / fs)
    mask   = (freqs >= BP_LOW) & (freqs <= BP_HIGH)
    n_segs = max(1, N // NPERSEG)
    psd_sum = torch.zeros(NPERSEG // 2 + 1, dtype=torch.float32)
    valid   = 0
    for i in range(n_segs):
        seg = cleaned[i * NPERSEG: (i + 1) * NPERSEG]
        if seg.shape[0] < NPERSEG:
            break
        fft_v    = torch.fft.rfft(seg)
        psd_sum += (fft_v.real**2 + fft_v.imag**2) / NPERSEG
        valid   += 1
    if valid == 0:
        return torch.tensor(0.0)
    return torch.mean(torch.abs((psd_sum / valid)[mask] - ref_psd[mask]) / (ref_psd[mask] + 1e-20))
