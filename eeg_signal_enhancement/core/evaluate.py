import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch, savgol_filter, medfilt

from config import FS_EEG, BP_LOW, BP_HIGH, SG_WIN, SG_POLY


def post_process(signal):
    """Clip outliers, apply median filter, then Savitzky-Golay smoothing."""
    p_lo, p_hi = np.percentile(signal, 2), np.percentile(signal, 98)
    sig = medfilt(np.clip(signal, p_lo, p_hi), kernel_size=5)
    w   = min(SG_WIN, len(sig))
    if w % 2 == 0:
        w -= 1
    return savgol_filter(sig, w, SG_POLY)


def eval_losses(cleaned, baseline, fs=FS_EEG):
    """Compute temporal RMS loss and frequency-domain PSD loss against the quiet baseline."""
    rms    = lambda x: float(np.sqrt(np.mean(x**2)))
    t_loss = abs(rms(cleaned) - rms(baseline)) / (rms(baseline) + 1e-10)
    nperseg = min(1024, min(len(cleaned), len(baseline)) // 4)
    f, pc  = welch(cleaned,  fs=fs, nperseg=nperseg)
    _, pb  = welch(baseline, fs=fs, nperseg=nperseg)
    band   = (f >= BP_LOW) & (f <= BP_HIGH)
    f_loss = float(np.mean(np.abs(pc[band] - pb[band]) / (pb[band] + 1e-20)))
    return {
        "temporal_loss":  t_loss,
        "frequency_loss": f_loss,
        "rms_cleaned":    rms(cleaned),
        "rms_baseline":   rms(baseline),
    }


def plot_results(eeg_noisy_eval, eeg_cleaned, eeg_baseline, title, save_path, fs=FS_EEG):
    """Save a two-panel time-series figure: full-scale and zero-centred comparison."""
    N_v  = min(int(10.0 * fs), len(eeg_noisy_eval), len(eeg_cleaned))
    t    = np.arange(N_v) / fs
    N_bl = min(N_v, len(eeg_baseline))

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(title, fontsize=12)

    ax = axes[0]
    ax.plot(t, eeg_noisy_eval[:N_v], color="tomato",    lw=1.5, alpha=0.85, label="Noisy")
    ax.plot(t, eeg_cleaned[:N_v],    color="steelblue", lw=1.5, alpha=0.9,  label="Cleaned")
    ax.plot(t[:N_bl], eeg_baseline[:N_bl] - np.mean(eeg_baseline[:N_bl]),
            color="seagreen", lw=1.5, alpha=0.9, label="Baseline")
    ax.set_ylim(np.percentile(eeg_noisy_eval[:N_v], 0.5),
                np.percentile(eeg_noisy_eval[:N_v], 99.5))
    ax.set_ylabel("Amplitude (uV)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Noisy / Cleaned / Baseline")
    ax.legend(loc="upper right", fontsize=10)

    cl_plot = eeg_cleaned[:N_v]   - np.mean(eeg_cleaned[:N_v])
    bl_plot = eeg_baseline[:N_bl] - np.mean(eeg_baseline[:N_bl])
    ax2 = axes[1]
    ax2.plot(t, cl_plot,        color="steelblue", lw=1.5, alpha=0.9, label="Cleaned")
    ax2.plot(t[:N_bl], bl_plot, color="seagreen",  lw=1.5, alpha=0.9, label="Baseline")
    ax2.set_ylabel("Amplitude (uV)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Cleaned vs Baseline (zero-centred)")
    ax2.legend(loc="upper right", fontsize=10)

    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [saved] {save_path}")
