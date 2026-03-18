import os
import time
import requests
import json
import base64
from typing import Optional, List, Dict, Any

class GeminiClient:
    """
    Standardized client for interacting with the Gemini API.
    Supports both text and image (multimodal) inputs.
    """
    def __init__(self, api_key: str, model: str = "gemini-3.0-pro"):
        self.api_key = api_key
        self.model = model
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

    def generate_content(self, prompt: str, image_b64: Optional[str] = None, max_retries: int = 5) -> str:
        """
        Sends a single prompt (with optional single image) to Gemini.
        """
        parts = [{"text": prompt}]
        if image_b64:
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_b64
                }
            })
        return self.generate_content_multi(parts, max_retries)

    def generate_content_multi(self, parts: List[Dict[str, Any]], max_retries: int = 5) -> str:
        """
        Generic method to send multiple parts (text/images) to Gemini.
        """
        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.2,
                "topK": 40,
                "topP": 0.8,
                "maxOutputTokens": 2048,
            }
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        content_parts = candidates[0].get("content", {}).get("parts", [])
                        for part in content_parts:
                            if "text" in part:
                                return part["text"]
                    return "Error: API returned no text content."
                
                if response.status_code in [429, 503]:
                    wait_time = 2 ** attempt
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                return f"Error: API status {response.status_code}. Details: {response.text}"
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Error: Request failed. Details: {str(e)}"
                time.sleep(2 ** attempt)
        
        return "Error: Unknown failure."

    def encode_image(self, image_path: str) -> str:
        """Encodes an image file to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
