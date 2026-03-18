import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import json

# Add core to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.llm_client import GeminiClient
from core.data_utils import load_csv_data, segment_data, save_json_results

class IMUAnalyzer:
    """
    Analyzes IMU modality using 5-point sensor integration and LLM inference.
    """
    def __init__(self, api_key: str, model: str = "gemini-3.0-pro"):
        self.client = GeminiClient(api_key, model)
        self.locations = [
            "head", "left_wrist", "right_wrist",
            "left_ankle", "right_ankle"
        ]
        self.sampling_rate = 100.0 # Hz (assumed)

    def integrate_sensors(self, data_root: str, window_idx: int) -> Dict[str, pd.DataFrame]:
        """
        Loads and synchronizes 8 IMU sensors for a given time window.
        Assumes data is stored in folders named after locations.
        """
        sensor_data = {}
        for loc in self.locations:
            path = os.path.join(data_root, loc, f"segment_{window_idx:03d}.csv")
            if os.path.exists(path):
                sensor_data[loc] = pd.read_csv(path)
            else:
                # Fallback to empty DF if missing
                sensor_data[loc] = pd.DataFrame()
        return sensor_data

    def plot_waveforms(self, sensor_data: Dict[str, pd.DataFrame], output_path: str):
        """Generates a visualization of the synchronized 5-sensor movements."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 9), sharex=True)
        axes = axes.flatten()
        
        for i, loc in enumerate(self.locations):
            df = sensor_data.get(loc)
            ax = axes[i]
            if df is not None and not df.empty:
                # Plot Magnitude of Acceleration (sqrt(ax^2 + ay^2 + az^2))
                acc_cols = [c for c in df.columns if 'acc' in c.lower() or 'a' in c.lower() and len(c) <= 2]
                if len(acc_cols) >= 3:
                    acc_mag = np.sqrt(np.sum(df[acc_cols[:3]]**2, axis=1))
                    ax.plot(acc_mag, label='Acc Mag', color='blue')
                
                # Plot Magnitude of Gyroscope
                gyro_cols = [c for c in df.columns if 'gyro' in c.lower() or 'g' in c.lower() and len(c) <= 2]
                if len(gyro_cols) >= 3:
                    gyro_mag = np.sqrt(np.sum(df[gyro_cols[:3]]**2, axis=1))
                    ax2 = ax.twinx()
                    ax2.plot(gyro_mag, label='Gyro Mag', color='red', alpha=0.5)
                    
            ax.set_title(f"Sensor: {loc.upper()}")
            ax.grid(True, alpha=0.3)
            
        axes[-1].set_visible(False)  # Hide unused 6th subplot
        plt.suptitle("5-Point IMU Integrated Dynamics")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path)
        plt.close()

    def build_prompt(self, imu_summary: str) -> str:
        """Constructs the prompt for IMU behavioral analysis."""
        prompt = f"""
You are an expert in biomechanics and human activity recognition (HAR). Analyze the following movement profile aggregated from 5 sensors (Head, Left/Right Wrist, Left/Right Ankle).

Motion Cues:
- Head Stability vs Limb Motion: Distinguishes between walking, standing, and reaching.
- Periodic Patterns in Ankles: Characteristic of gait, stair climbing, or cycling.
- High Dynamic Range in Wrists: Indicates active interactions (e.g., waving, obstacle avoidance).
- Stability across all sensors: Suggests static posture or focused observation.

Aggregated Motion Profile:
{imu_summary}

Based on this kinetic data, infer the following three dimensions:
1. Scene: The environmental context (e.g., stairs, flat ground, workspace).
2. Action: The specific motor action or posture (e.g., "walking forward", "reaching for object").
3. Awareness: The attentional state inferred from movement (e.g., "purposeful navigation", "exploratory scanning").

Return in this JSON format:
{{
  "Scene": "<string>",
  "Action": "<string>",
  "Awareness": "<string>"
}}
"""
        return prompt

    def analyze(self, data_root: str, output_dir: str):
        """Full pipeline for IMU."""
        print(f"Starting IMU analysis using data from {data_root}...")
        os.makedirs(output_dir, exist_ok=True)
        
        # We assume 1-second segments are already provided or managed by the caller
        # For simplicity, let's assume there are 10 segments to process
        all_results = []
        
        for i in range(10): # Example limit
            print(f"Integrating sensors for segment {i}...")
            sensor_data = self.integrate_sensors(data_root, i)
            
            # 1. Feature Visualization
            vis_path = os.path.join(output_dir, f"segment_{i:03d}_imu.png")
            self.plot_waveforms(sensor_data, vis_path)
            
            # 2. Summary for Prompt (statistical summary)
            summary_parts = []
            for loc, df in sensor_data.items():
                if not df.empty:
                    mag_mean = np.mean(np.sqrt(np.sum(df.select_dtypes(include=np.number).iloc[:,:3]**2, axis=1)))
                    summary_parts.append(f"{loc}: Mean Acc Mag = {mag_mean:.2f}")
            imu_summary = "\n".join(summary_parts)
            
            # 3. LLM Inference
            prompt = self.build_prompt(imu_summary)
            image_b64 = self.client.encode_image(vis_path)
            response = self.client.generate_content(prompt, image_b64)
            
            # 4. Parse Response
            try:
                clean_response = response.strip().replace('```json', '').replace('```', '')
                result = json.loads(clean_response)
            except:
                result = {"raw_response": response, "error": "Failed to parse JSON"}
            
            all_results.append({
                "segment_index": i,
                "analysis": result
            })
            
        output_json = os.path.join(output_dir, "imu_analysis_results.json")
        save_json_results({"metadata": {"source": data_root, "type": "imu"}, "results": all_results}, output_json)
        print(f"IMU analysis complete. Saved to {output_json}")

if __name__ == "__main__":
    API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY")
    analyzer = IMUAnalyzer(API_KEY)
    # Example: analyzer.analyze("./data/imu_samples", "./outputs/imu_test")
