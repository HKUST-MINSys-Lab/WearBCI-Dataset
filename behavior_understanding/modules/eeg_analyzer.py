import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Add core to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.llm_client import GeminiClient
from core.data_utils import load_csv_data, segment_data, save_json_results

class EEGAnalyzer:
    """
    Analyzes EEG modality using PSD features and LLM inference.
    """
    def __init__(self, api_key: str, model: str = "gemini-3.0-pro"):
        self.client = GeminiClient(api_key, model)
        self.channels = ["Fp1", "Fp2", "F7", "F8", "T3", "T4", "O1", "O2"]
        self.bands = {
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30)
        }
        self.sampling_rate = 250.0 # Hz (assumed based on previous code)

    def calculate_psd(self, segment: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculates PSD for each channel and frequency band."""
        psd_results = {}
        for channel in self.channels:
            if channel not in segment.columns:
                continue
            
            data = segment[channel].values
            # Compute PSD using Welch's method
            freqs, psd = welch(data, fs=self.sampling_rate, nperseg=min(len(data), 256))
            
            band_powers = {}
            for band_name, (low, high) in self.bands.items():
                mask = (freqs >= low) & (freqs < high)
                power = np.trapz(psd[mask], freqs[mask]) if any(mask) else 0.0
                band_powers[band_name] = float(power)
            
            psd_results[channel] = band_powers
            
        return psd_results

    def plot_psd(self, psd_data: Dict[str, Dict[str, float]], output_path: str):
        """Generates a visualization of PSD features."""
        plt.figure(figsize=(10, 6))
        channels = list(psd_data.keys())
        bands = list(self.bands.keys())
        
        # Prepare data for plotting
        x = np.arange(len(channels))
        width = 0.2
        
        for i, band in enumerate(bands):
            powers = [psd_data[ch][band] for ch in channels]
            plt.bar(x + i*width, powers, width, label=band.capitalize())
            
        plt.xlabel('Channels')
        plt.ylabel('Relative Power')
        plt.title('EEG Power Spectral Density by Channel and Band')
        plt.xticks(x + width, channels)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def build_prompt(self, psd_summary: Dict[str, Any]) -> str:
        """Constructs the prompt for EEG behavioral analysis."""
        prompt = f"""
You are an expert neuroscientist and behavioral analyst. Analyze the following EEG Power Spectral Density (PSD) data recorded from 8 dry electrodes (Fp1, Fp2, F7, F8, T3, T4, O1, O2).

Neural Cues for Navigation Context:
- Decreased Alpha (Fp1, Fp2, O1, O2) + Increased Beta: Focused visual attention or cognitive engagement.
- Increased Theta (T3, T4): Memory retrieval, spatial planning, or high cognitive load.
- Asymmetry (F7 vs F8): Potential emotional or affective bias.
- Frontal Alpha Increase: Relaxation or reduced external focus.

PSD Data (Relative Power):
{psd_summary}

Based on this neural activity, infer the following three dimensions of human behavior:
1. Scene: The likely environmental context.
2. Action: The probable motor behavior or posture.
3. Awareness: The mental engagement or attentional state.

Return exactly in this JSON format:
{{
  "Scene": "<string>",
  "Action": "<string>",
  "Awareness": "<string>"
}}
"""
        return prompt

    def analyze(self, csv_path: str, output_dir: str):
        """Full pipeline: Load -> Segment -> Feature Extraction -> LLM -> Save."""
        print(f"Loading EEG data from {csv_path}...")
        df = load_csv_data(csv_path)
        segments = segment_data(df, window_size_sec=1.0, sampling_rate=self.sampling_rate)
        
        print(f"Segmented into {len(segments)} windows. Starting analysis...")
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = []
        
        # Limit processing to first 10 for demonstration/testing if needed, 
        # but here we follow the user request for a complete library.
        for i, segment in enumerate(segments):
            print(f"Processing segment {i+1}/{len(segments)}...")
            
            # 1. Feature Extraction
            psd_data = self.calculate_psd(segment)
            
            # 2. Visualization
            vis_path = os.path.join(output_dir, f"segment_{i:03d}_psd.png")
            self.plot_psd(psd_data, vis_path)
            
            # 3. LLM Inference
            prompt = self.build_prompt(psd_data)
            image_b64 = self.client.encode_image(vis_path)
            response = self.client.generate_content(prompt, image_b64)
            
            # 4. Parse Response
            try:
                # Basic cleaning of LLM response if it contains markdown markers
                clean_response = response.strip().replace('```json', '').replace('```', '')
                result = json.loads(clean_response)
            except:
                result = {"raw_response": response, "error": "Failed to parse JSON"}
            
            # Add metadata
            entry = {
                "segment_index": i,
                "timestamp_start": float(i),
                "timestamp_end": float(i + 1),
                "analysis": result
            }
            all_results.append(entry)
            
            # Basic rate limit mitigation
            import time
            time.sleep(1)
            
        # 5. Save aggregate results
        output_json = os.path.join(output_dir, "eeg_analysis_results.json")
        save_json_results({"metadata": {"source": csv_path, "type": "eeg"}, "results": all_results}, output_json)
        print(f"EEG analysis complete. Results saved to {output_json}")

if __name__ == "__main__":
    import json
    # For testing, you would set your API key in env or pass it here
    API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY")
    analyzer = EEGAnalyzer(API_KEY)
    
    # Example usage:
    # analyzer.analyze("./data/eeg/1.csv", "./outputs/eeg_test")
