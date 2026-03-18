import os
import sys
import json
import base64
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Add core to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.llm_client import GeminiClient
from core.data_utils import save_json_results

class FusionAnalyzer:
    """
    Implements both Concatenate Fusion (early/pre-fusion) and Optimized Fusion (late/post-fusion).
    """
    def __init__(self, api_key: str, model: str = "gemini-3.0-pro"):
        self.client = GeminiClient(api_key, model)

    def build_concatenate_prompt(self, eeg_summary: str, imu_summary: str) -> str:
        """Constructs a unified prompt for Concatenate (Pre-fusion)."""
        prompt = f"""
You are a high-level multimodal behavioral analyst. You are provided with synchronized data from three sources:
1. Video: Egocentric RGB view from a chest camera.
2. EEG: Neural activity (PSD) summarizing cognitive engagement.
3. IMU: Kinematic data from 5 sensors (Head, Left/Right Wrist, Left/Right Ankle) tracking motor dynamics.

Contextual Guidance:
- Video grounds the environmental context (Scene).
- IMU tracks movement intensity and motor patterns (Action).
- EEG reflects attention, workload, and cognitive states (Cognition/Awareness).

Synchronized Summaries:
- EEG Metrics: {eeg_summary}
- IMU Metrics: {imu_summary}

Task (Concatenate Fusion): Use the visual evidence in the images and the provided metrics to reconcile any conflicts and provide a unified, multi-dimensional understanding of human behavior.

Return exactly in this JSON format:
{{
  "Scene": "<unified scene description>",
  "Action": "<unified motor behavior description>",
  "Awareness": "<unified cognitive/attentional state>"
}}
"""
        return prompt

    def build_optimized_fusion_prompt(self, video_out: str, eeg_out: str, imu_out: str) -> str:
        """Constructs a prompt for Optimized Fusion (Post-fusion/Late Fusion)."""
        prompt = f"""
You are a master behavioral integrator. You are provided with three independent interpretations of a human behavior event from different sensor modalities:

1. Video-Only Interpretation: {video_out}
2. EEG-Only Interpretation: {eeg_out}
3. IMU-Only Interpretation: {imu_out}

Task (Optimized Fusion): Your goal is to apply "Optimized Fusion" by:
- Estimating the reliability of each modality (e.g., Video is best for Scene, IMU for Action, EEG for Cognition).
- Resolving contradictions between reports (e.g., if IMU detects walking but Video says standing, check if Video is looking at a static object).
- Calibrating the final result to reconcile omissions and conflicts.

Provide the final, highest-precision multi-dimensional behavioral understanding.

Return exactly in this JSON format:
{{
  "Scene": "<calibrated scene description>",
  "Action": "<calibrated action description>",
  "Awareness": "<calibrated cognition/awareness description>"
}}
"""
        return prompt

    def concatenate_fusion(self, 
                          video_frame_path: str, 
                          eeg_psd_path: str, 
                          imu_wave_path: str,
                          eeg_metrics: Dict[str, Any],
                          imu_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs Concatenate Fusion (Early/Pre-fusion) by combining raw features into one prompt.
        """
        # Encode images
        images = []
        for path in [video_frame_path, eeg_psd_path, imu_wave_path]:
            if os.path.exists(path):
                images.append(self.client.encode_image(path))
            else:
                images.append(None)
                
        # Build Prompt
        prompt = self.build_concatenate_prompt(json.dumps(eeg_metrics), json.dumps(imu_metrics))
        
        parts = [{"text": prompt}]
        for img_b64 in images:
            if img_b64:
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})
        
        try:
            r = self.client.generate_content_multi(parts)
            clean_res = r.strip().replace('```json', '').replace('```', '')
            result = json.loads(clean_res)
        except:
            result = {"error": "Concatenate fusion failed", "raw": r if 'r' in locals() else "None"}
        return result

    def optimized_fusion(self, video_json: Dict, eeg_json: Dict, imu_json: Dict) -> Dict[str, Any]:
        """
        Performs Optimized Fusion (Late/Post-fusion) by reconciling textual outputs.
        """
        prompt = self.build_optimized_fusion_prompt(
            video_out=json.dumps(video_json.get("analysis", {})),
            eeg_out=json.dumps(eeg_json.get("analysis", {})),
            imu_out=json.dumps(imu_json.get("analysis", {}))
        )
        
        try:
            r = self.client.generate_content(prompt)
            clean_res = r.strip().replace('```json', '').replace('```', '')
            result = json.loads(clean_res)
        except:
            result = {"error": "Optimized fusion failed", "raw": r if 'r' in locals() else "None"}
        return result

    def batch_fusion_analysis(self, video_dir: str, eeg_dir: str, imu_dir: str, output_dir: str, method: str = "concatenate"):
        """Processes all modalities in parallel and fuses results using the specified method."""
        os.makedirs(output_dir, exist_ok=True)
        all_results = []
        
        # Load independent results for Optimized Fusion if needed
        video_results_path = os.path.join(video_dir, "video_analysis_results.json")
        eeg_results_path = os.path.join(eeg_dir, "eeg_analysis_results.json")
        imu_results_path = os.path.join(imu_dir, "imu_analysis_results.json")

        video_data = []
        eeg_data = []
        imu_data = []

        if method == "optimized":
            if all(os.path.exists(p) for p in [video_results_path, eeg_results_path, imu_results_path]):
                with open(video_results_path, 'r') as f: video_data = json.load(f).get("results", [])
                with open(eeg_results_path, 'r') as f: eeg_data = json.load(f).get("results", [])
                with open(imu_results_path, 'r') as f: imu_data = json.load(f).get("results", [])
            else:
                print("Error: Single modality results missing for Optimized Fusion.")
                return

        # Process windows (assuming synchronized indices)
        num_windows = 10 # Example limit
        for i in range(num_windows):
            print(f"Running {method} fusion for window {i}...")
            
            if method == "concatenate":
                v_frame = os.path.join(video_dir, "frames", f"frame_{i:04d}_t{i}s.jpg")
                e_psd = os.path.join(eeg_dir, f"segment_{i:03d}_psd.png")
                i_wave = os.path.join(imu_dir, f"segment_{i:03d}_imu.png")
                # Mock metrics for structure
                e_metrics = {"notes": "Neural engagement context"}
                i_metrics = {"notes": "Motor dynamics context"}
                
                result = self.concatenate_fusion(v_frame, e_psd, i_wave, e_metrics, i_metrics)
            
            elif method == "optimized":
                # Find matching segments by index
                v_item = next((x for x in video_data if x["segment_index"] == i), {})
                e_item = next((x for x in eeg_data if x["segment_index"] == i), {})
                i_item = next((x for x in imu_data if x["segment_index"] == i), {})
                
                result = self.optimized_fusion(v_item, e_item, i_item)

            all_results.append({
                "window": i,
                f"{method}_result": result
            })
            time.sleep(1) # Basic rate limit mitigation
            
        output_file = f"{method}_fusion_results.json"
        save_json_results({"metadata": {"strategy": method}, "results": all_results}, os.path.join(output_dir, output_file))
        print(f"Fusion ({method}) complete. Saved to {output_file}")

if __name__ == "__main__":
    API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY")
    analyzer = FusionAnalyzer(API_KEY)
    # analyzer.batch_fusion_analysis("./outputs/video_test", "./outputs/eeg_test", "./outputs/imu_test", "./outputs/fusion_test")
