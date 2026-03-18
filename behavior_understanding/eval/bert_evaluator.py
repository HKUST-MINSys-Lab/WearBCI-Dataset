import os
import sys
import json
import torch
from typing import List, Dict, Any
from bert_score import score

# Add core to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class BERTEvaluator:
    """
    Evaluates semantic similarity between LLM-generated behavioral descriptions
    and Ground Truth annotations using BERTScore F1.
    """
    def __init__(self, model_type: str = 'bert-base-uncased'):
        self.model_type = model_type

    def load_json(self, path: str) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def align_and_evaluate(self, candidate_path: str, gt_path: str) -> Dict[str, Any]:
        """
        Aligns 1s analyzer output with 10s Ground Truth slots and computes scores.
        """
        print(f"Loading files for evaluation...")
        candidate_data = self.load_json(candidate_path)
        gt_data = self.load_json(gt_path)
        
        candidates = candidate_data.get("results", [])
        gt_timeline = gt_data.get("timeline", [])
        
        evaluation_results = []
        
        # Semantic categories for scoring
        categories = ["Scene", "Action", "Awareness"] # "Awareness" is Cognition in paper
        
        # Prepare lists for BERTScore
        # candidate_texts[cat] = [list of strings]
        # reference_texts[cat] = [list of strings]
        scoring_batches = {cat: {"cands": [], "refs": []} for cat in categories}

        for cand in candidates:
            # Assumes cand has 'timestamp_seconds' or 'segment_index' (treated as seconds)
            ts = cand.get("timestamp_seconds", cand.get("segment_index", 0))
            
            # Find matching GT slot
            matching_gt = next((slot for slot in gt_timeline 
                                if slot["start_seconds"] <= ts < slot["end_seconds"]), None)
            
            if matching_gt:
                analysis = cand.get("analysis", {})
                for cat in categories:
                    cand_text = str(analysis.get(cat, ""))
                    # Map Awareness to cognition in GT template if needed, 
                    # but here we assume labels match or we handle mapping
                    ref_key = "cognition" if cat == "Awareness" else cat.lower()
                    ref_text = str(matching_gt.get(ref_key, ""))
                    
                    if cand_text and ref_text:
                        scoring_batches[cat]["cands"].append(cand_text)
                        scoring_batches[cat]["refs"].append(ref_text)

        # Compute BERTScores
        final_scores = {}
        overall_f1 = 0
        valid_cats = 0
        
        print("Computing BERTScores (this may take a moment)...")
        for cat, batch in scoring_batches.items():
            if not batch["cands"]:
                final_scores[cat] = 0.0
                continue
            
            # P, R, F1 = score(cands, refs, lang="en", verbose=False, model_type=self.model_type)
            # Default to English and CPU/GPU as available
            P, R, F1 = score(batch["cands"], batch["refs"], lang="en", verbose=False)
            
            avg_f1 = float(F1.mean())
            final_scores[cat] = round(avg_f1, 4)
            overall_f1 += avg_f1
            valid_cats += 1
            
        final_scores["Overall"] = round(overall_f1 / valid_cats, 4) if valid_cats > 0 else 0.0
        
        return {
            "metadata": {
                "candidate_source": candidate_path,
                "gt_source": gt_path,
                "model": "bert-score default"
            },
            "scores": final_scores
        }

if __name__ == "__main__":
    evaluator = BERTEvaluator()
    # Example usage:
    # results = evaluator.align_and_evaluate("./outputs/fusion_test/multimodal_fusion_results.json", "./data/templates/ground_truth.json")
    # print(json.dumps(results, indent=2))
