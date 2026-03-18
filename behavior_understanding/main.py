import os
import argparse
from modules.eeg_analyzer import EEGAnalyzer
from modules.imu_analyzer import IMUAnalyzer
from modules.video_analyzer import VideoAnalyzer
from modules.fusion_analyzer import FusionAnalyzer
from eval.bert_evaluator import BERTEvaluator

def main():
    parser = argparse.ArgumentParser(description="WearBCI: Multi-modal Behavior Understaning Pipeline")
    parser.add_argument("--api_key", required=True, help="Gemini API Key")
    parser.add_argument("--mode", choices=["eeg", "imu", "video", "concatenate", "optimized", "eval", "all"], default="all")
    parser.add_argument("--eeg_csv", default="./data/eeg/1.csv")
    parser.add_argument("--imu_root", default="./data/imu")
    parser.add_argument("--video_path", default="./data/video/1.mov")
    parser.add_argument("--gt_path", default="./data/templates/ground_truth.json")
    parser.add_argument("--output_dir", default="./outputs")
    
    args = parser.parse_args()
    api_key = args.api_key
    
    if args.mode in ["eeg", "all"]:
        print("\n--- Running EEG Analysis ---")
        analyzer = EEGAnalyzer(api_key)
        analyzer.analyze(args.eeg_csv, os.path.join(args.output_dir, "eeg"))
        
    if args.mode in ["imu", "all"]:
        print("\n--- Running IMU Analysis ---")
        analyzer = IMUAnalyzer(api_key)
        analyzer.analyze(args.imu_root, os.path.join(args.output_dir, "imu"))
        
    if args.mode in ["video", "all"]:
        print("\n--- Running Video Analysis ---")
        analyzer = VideoAnalyzer(api_key)
        analyzer.analyze(args.video_path, os.path.join(args.output_dir, "video"))
        
    if args.mode in ["concatenate", "optimized", "all"]:
        method = "optimized" if args.mode == "optimized" else "concatenate"
        print(f"\n--- Running Multimodal Fusion ({method}) ---")
        analyzer = FusionAnalyzer(api_key)
        # For optimized fusion, single modality results must exist
        analyzer.batch_fusion_analysis(
            os.path.join(args.output_dir, "video"),
            os.path.join(args.output_dir, "eeg"),
            os.path.join(args.output_dir, "imu"),
            os.path.join(args.output_dir, "fusion"),
            method=method
        )

    if args.mode in ["eval", "all"]:
        print("\n--- Running BERTScore Evaluation ---")
        evaluator = BERTEvaluator()
        # Evaluate optimized fusion by default if running 'all'
        fusion_file = "optimized_fusion_results.json" if args.mode in ["optimized", "all"] else "concatenate_fusion_results.json"
        result_path = os.path.join(args.output_dir, "fusion", fusion_file)
        if os.path.exists(result_path):
            scores = evaluator.align_and_evaluate(result_path, args.gt_path)
            print("\nEvaluation Results:")
            print(scores["scores"])
        else:
            print(f"Warning: Fusion results not found at {result_path}. Skipping evaluation.")

if __name__ == "__main__":
    main()
