import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

def compute_time_metrics_asymmetric(
    pred_time: float,
    gt_answer_time: float,
    gt_question_time: float,
    alpha: float = 0.5,     # smaller alpha so the late-response penalty is less steep
    tolerance: float = 0.0  # tolerance in seconds (accounts for FPS and processing time)
) -> Dict[str, float]:
    """
    Asymmetric time-based score:
    - Predicted before the question is fully asked: 0
    - After the question but before the event (beyond tolerance): 0 (hallucination)
    - Near the ground-truth event (within tolerance): 1
    - After the event: exponential decay
    """

    # 1. Hard-invalid range: answered before the question ends.
    if pred_time < gt_question_time:
        return {
            "deviation": pred_time - gt_answer_time,
            "status": "invalid_pre_question",  # answered too early / invalid
            "score": 0.0
        }

    deviation = pred_time - gt_answer_time

    # 2. Hallucination range: answered before the event (outside tolerance).
    # Example: the ball lands at 20s but the model predicts 15s — wrong.
    if deviation < -tolerance:
        return {
            "deviation": deviation,
            "status": "premature_hallucination",  # hallucination
            "score": 0.0
        }

    # 3. Perfect range: near the ground-truth event.
    if abs(deviation) <= tolerance:
        return {
            "deviation": deviation,
            "status": "perfect",
            "score": 1.0
        }

    # 4. Late range: after the ground-truth event.
    # Effective delay (subtract tolerance to smooth the decay curve).
    effective_delay = deviation - tolerance
    score = math.exp(-alpha * effective_delay)
    
    return {
        "deviation": deviation,
        "status": "late",
        "score": score
    }

def evaluate_predictions(predictions_file: str, alpha: float = 2.0) -> None:
    predictions_path = Path(predictions_file)
    if not predictions_path.exists():
        raise FileNotFoundError(f"File not found: {predictions_file}")
    
    print(f"Loading: {predictions_file}")
    data = []
    
    # Accept both JSON and JSONL.
    try:
        with open(predictions_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Try to parse the whole file as a JSON list first.
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Fall back to JSONL parsing (one JSON per line).
                data = [json.loads(line) for line in content.splitlines() if line.strip()]
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    metrics_list = []
    
    for i, item in enumerate(data):
        pred = item.get("prediction_answer_time")
        gt_ans = item.get("gt_answer_time")
        gt_ques = item.get("gt_question_time")

        if pred is None or gt_ans is None or gt_ques is None:
            if pred is None or gt_ans is None: continue
            if gt_ques is None: gt_ques = 0.0 
        
        try:
            pred = float(pred)
            gt_ans = float(gt_ans)
            gt_ques = float(gt_ques)
        except:
            continue
            
        m = compute_time_metrics_asymmetric(pred, gt_ans, gt_ques, alpha=alpha, tolerance=0.0)
        metrics_list.append(m)

    if not metrics_list:
        print("No valid data found.")
        return

    # === Statistics ===
    scores = [m["score"] for m in metrics_list]
    deviations = [m["deviation"] for m in metrics_list]

    # Status counts
    status_counts = {"perfect": 0, "late": 0, "premature_hallucination": 0, "invalid_pre_question": 0}
    for m in metrics_list:
        status_counts[m["status"]] = status_counts.get(m["status"], 0) + 1

    print("\n" + "="*60)
    print("Stream3D Evaluation Results (Asymmetric Logic)")
    print("="*60)
    print(f"Total Samples: {len(metrics_list)}")
    print(f"Mean Score:    {np.mean(scores):.4f}")
    print("-" * 60)
    print("Status Distribution:")
    for status, count in status_counts.items():
        pct = count / len(metrics_list) * 100
        print(f"  - {status:<25}: {count:4d} ({pct:.2f}%)")
    print("-" * 60)
    print("Time Deviation (Pred - GT):")
    print(f"  - Mean:   {np.mean(deviations):.4f}s (Negative means early)")
    print(f"  - Median: {np.median(deviations):.4f}s")
    print("="*60)

if __name__ == "__main__":
    input_file = "output_logs/Stream3D-VLM-4B/evaluate_results.json"
    alpha = 0.5
    evaluate_predictions(predictions_file=input_file, alpha=alpha)