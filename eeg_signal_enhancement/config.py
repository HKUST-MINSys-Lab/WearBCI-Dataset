import os

HERE        = os.path.dirname(os.path.abspath(__file__))
EEG_DIR     = os.path.join(HERE, "data", "eeg")
IMU_DIR     = os.path.join(HERE, "data", "imu")
RESULTS_DIR = os.path.join(HERE, "results")

# EEG channel to process
CHANNEL = "Channel8"

# IMU sensor placements
IMU_PARTS = ["head", "left_wrist", "right_wrist", "left_ankle", "right_ankle"]

# Walking task files
WALKING_FILES = ["walking_slow_2.csv", "walking_medium_3.csv", "walking_fast_4.csv"]

# Signal processing
FS_EEG   = 250    # EEG sampling rate (Hz)
BP_LOW   = 1.0    # Bandpass low cutoff (Hz)
BP_HIGH  = 45.0   # Bandpass high cutoff (Hz)
NPERSEG  = 256    # FFT segment length for PSD

# Windowing
WIN_SEC  = 2.0    # Window size (s)
HOP_SEC  = 1.0    # Hop size (s)
LAG_MS   = 250    # ±IMU lag range (ms)
LAG_STEP = 15     # Lag step (ms)

# Training
TRAIN_RATIO    = 0.70
RIDGE_ALPHA    = 100
FINETUNE_STEPS = 30
LR             = 3e-3
LAMBDA_T       = 0.5   # Temporal loss weight
LAMBDA_F       = 0.5   # Frequency loss weight
LAMBDA_R       = 1e-3  # L2 regularization weight

# Post-processing (Savitzky-Golay)
SG_WIN  = 51
SG_POLY = 3
