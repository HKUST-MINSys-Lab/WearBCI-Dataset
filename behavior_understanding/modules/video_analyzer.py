import os
import sys
import cv2
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Add core to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.llm_client import GeminiClient
from core.data_utils import save_json_results

class VideoAnalyzer:
    """
    Analyzes egocentric video modality using 1 FPS frame extraction and LLM inference.
    """
    def __init__(self, api_key: str, model: str = "gemini-3.0-pro"):
        self.client = GeminiClient(api_key, model)
        self.fps_target = 1.0

    def extract_frames(self, video_path: str, output_dir: str) -> List[str]:
        """Extracts frames at 1 FPS from the given video."""
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(orig_fps / self.fps_target)
        
        frame_paths = []
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_name = f"frame_{saved_count:04d}_t{saved_count}s.jpg"
                frame_path = os.path.join(output_dir, frame_name)
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                saved_count += 1
                
            frame_count += 1
            
        cap.release()
        print(f"Extracted {saved_count} frames from {video_path}")
        return frame_paths

    def build_prompt(self) -> str:
        """Constructs the prompt for Video-based behavioral analysis."""
        prompt = """
You are an expert in behavioral observation and scene understanding. Analyze this egocentric (first-person) RGB image captured from a chest camera during navigation.

Visual Cues:
- Environmental Layout: Identify room types, lighting, and salient objects.
- Perspective Geometry: Infer the subject's relationship to the environment (standing near, approaching, turning).
- Contextual Clues: Infer probable intent or focus from objects and affordances.

Based on this image, infer the following three dimensions:
1. Scene: The environmental context and layout.
2. Action: The probable body motion or posture.
3. Awareness: The likely cognitive or attentional state.

Return in this JSON format:
{
  "Scene": "<string>",
  "Action": "<string>",
  "Awareness": "<string>"
}
"""
        return prompt

    def analyze(self, video_path: str, output_dir: str):
        """Full pipeline for Video."""
        print(f"Starting Video analysis for {video_path}...")
        frames_dir = os.path.join(output_dir, "frames")
        frame_paths = self.extract_frames(video_path, frames_dir)
        
        all_results = []
        prompt = self.build_prompt()
        
        # Limit to first 10 for demonstration/testing
        for i, frame_path in enumerate(frame_paths[:10]):
            print(f"Analyzing frame {i+1}/{len(frame_paths[:10])}...")
            
            image_b64 = self.client.encode_image(frame_path)
            response = self.client.generate_content(prompt, image_b64)
            
            try:
                clean_response = response.strip().replace('```json', '').replace('```', '')
                result = json.loads(clean_response)
            except:
                result = {"raw_response": response, "error": "Failed to parse JSON"}
            
            all_results.append({
                "segment_index": i,
                "timestamp_seconds": float(i),
                "frame_path": frame_path,
                "analysis": result
            })
            
            # Rate limit mitigation
            import time
            time.sleep(1)
            
        output_json = os.path.join(output_dir, "video_analysis_results.json")
        save_json_results({"metadata": {"source": video_path, "type": "video"}, "results": all_results}, output_json)
        print(f"Video analysis complete. Saved to {output_json}")

if __name__ == "__main__":
    API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY")
    analyzer = VideoAnalyzer(API_KEY)
    # Example: analyzer.analyze("./data/video.mov", "./outputs/video_test")
