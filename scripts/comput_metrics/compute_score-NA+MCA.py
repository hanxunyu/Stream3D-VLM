import json
import numpy as np
import pandas as pd
from pathlib import Path

# ================== Configuration ==================
Backward = [
    "camera_direction_backward",
    "camera_rotation_backward",
    "object2camera_direction_backward",
    "object_counting_backward",
    "appearance_order_backward",
    "camera_displacement_backward",
    "camera_path_backward",
    "object2camera_distance_backward",
    "appearance_time_backward",
    "camera_comprehensive_backward",
    "object2camera_location_backward",
    "room_area_backward"
]

Realtime = [
    "object2camera_reldistance_realtime",
    "object2object_distance_realtime",
    "object2camera_distance_realtime",
    "object_size_realtime",
    "object_property_realtime",
    "object_position_realtime",
    "object_recognition_realtime"   
]

Forward = [
    "camera_direction_forward",
    "camera_rotation_forward",
    "object2camera_direction_forward",
    "camera_displacement_forward",
    "camera_path_forward",
    "object2camera_distance_forward",
    "camera_comprehensive_forward",
    "object2camera_location_forward",
    "object_finding_forward",
    "room_area_forward"

]

NA_QUESTION_TYPES = [
    "camera_displacement_backward",
    "camera_displacement_forward",
    "camera_path_backward",
    "camera_path_forward",
    "object2camera_distance_backward",
    "object2camera_distance_realtime",
    "object2camera_distance_forward",
    "object2object_distance_realtime",
    "appearance_time_backward"
]

MCA_QUESTION_TYPES = [
    "camera_direction_backward",
    "camera_direction_forward",
    "camera_rotation_backward",
    "camera_rotation_forward",
    "object2camera_direction_backward",
    "object2camera_direction_forward",
    "object_counting_backward",
    "appearance_order_backward",
    "object2camera_reldistance_realtime"
]

OEA_QUESTION_TYPES = [
    "camera_comprehensive_backward",
    "camera_comprehensive_forward",
    "object2camera_location_backward",
    "object2camera_location_forward",
    "room_area_backward",
    "room_area_forward",
    "object_finding_forward",
    "object_size_realtime",
    "object_property_realtime",
    "object_position_realtime",
    "object_recognition_realtime"
]
 
METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "mean_relative_accuracy",
}

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA:.5:.95:.05": 0.,
}

# ================== Helper functions ==================
def fuzzy_matching(pred):
    """Extract the first word from prediction and clean it"""
    if pred is None or pred == "":
        return ""
    # Handle potential non-string input
    return str(pred).split(' ')[0].rstrip('.').strip()

def exact_match(pred, target):
    """Calculate exact match accuracy"""
    return 1. if str(pred).lower() == str(target).lower() else 0.

def abs_dist_norm(pred, target):
    """Calculate normalized absolute distance"""
    if target == 0:
        return float('inf') if pred != 0 else 0
    return abs(pred - target) / abs(target)

def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    """Calculate mean relative accuracy across confidence intervals"""
    num_pts = int((end - start) / interval) + 1
    conf_intervs = np.linspace(start, end, num_pts)
    dist = abs_dist_norm(pred, target)
    accuracy = dist <= (1 - conf_intervs)
    return accuracy.mean()

def to_float(pred):
    """Convert prediction to float"""
    try:
        return float(pred)
    except (ValueError, TypeError):
        return None

# ================== Field extraction ==================
def evaluate_single_result(doc):
    """Evaluate a single prediction"""
    prediction = fuzzy_matching(doc.get('prediction', ''))
    
    # Use 'ground_truth' (renamed from 'GT_answer')
    ground_truth = doc.get('ground_truth', '')
    
    question_type = doc.get('question_type', '')
    
    result = {
        'question_type': question_type,
        'prediction': prediction,
        'ground_truth': ground_truth,
    }
    
    # Skip OEA question types
    if question_type in OEA_QUESTION_TYPES:
        result['skip'] = True
        return result
    
    # Determine if it's MCA or NA based on question_type
    if question_type in MCA_QUESTION_TYPES:
        # Multiple Choice Answer
        result['accuracy'] = exact_match(prediction, ground_truth)
        result['metric_type'] = 'MCA'
    elif question_type in NA_QUESTION_TYPES:
        # Numerical Answer
        pred_float = to_float(prediction)
        target_float = to_float(ground_truth)
        
        if pred_float is not None and target_float is not None:
            result['MRA:.5:.95:.05'] = mean_relative_accuracy(pred_float, target_float)
        else:
            result['MRA:.5:.95:.05'] = WORST_CASE_FOR_METRICS['MRA:.5:.95:.05']
        result['metric_type'] = 'NA'
    else:
        # Print a warning for unknown question types instead of raising
        print(f"Warning: Unknown question type: {question_type}")
        result['skip'] = True
    
    return result

def get_category(question_type):
    """Determine which category (Backward/Realtime/Forward) a question belongs to"""
    if question_type in Backward:
        return 'Backward'
    elif question_type in Realtime:
        return 'Realtime'
    elif question_type in Forward:
        return 'Forward'
    else:
        return 'Unknown'

def aggregate_results(results_df):
    """Aggregate results by category and metric type"""
    # Filter out skipped questions
    if 'skip' in results_df.columns:
        results_df = results_df[results_df['skip'] != True]
    
    # Add category column
    results_df['category'] = results_df['question_type'].apply(get_category)
    
    output = {}
    detailed_output = {}
    
    # Process each category
    for category in ['Backward', 'Realtime', 'Forward']:
        category_df = results_df[results_df['category'] == category]
        
        if len(category_df) == 0:
            continue
        
        detailed_output[category] = {}
        
        # MCA questions in this category
        mca_df = category_df[category_df['metric_type'] == 'MCA']
        if len(mca_df) > 0 and 'accuracy' in mca_df.columns:
            mca_scores = []
            for question_type in mca_df['question_type'].unique():
                qt_df = mca_df[mca_df['question_type'] == question_type]
                score = qt_df['accuracy'].mean()
                detailed_output[category][f"{question_type}_accuracy"] = score
                mca_scores.append(score)
            
            if mca_scores:
                output[f"{category}_MCA_avg"] = np.mean(mca_scores)
        
        # NA questions in this category
        na_df = category_df[category_df['metric_type'] == 'NA']
        if len(na_df) > 0 and 'MRA:.5:.95:.05' in na_df.columns:
            na_scores = []
            for question_type in na_df['question_type'].unique():
                qt_df = na_df[na_df['question_type'] == question_type]
                score = qt_df['MRA:.5:.95:.05'].mean()
                detailed_output[category][f"{question_type}_MRA"] = score
                na_scores.append(score)
            
            if na_scores:
                output[f"{category}_NA_avg"] = np.mean(na_scores)
    
    # Calculate overall averages
    all_scores = list(output.values())
    if all_scores:
        output['Overall'] = np.mean(all_scores)
    
    return output, detailed_output

# ================== File loading ==================
def evaluate_predictions(json_file):
    """Main evaluation function"""
    print(f"Loading predictions from {json_file}...")
    
    results = []
    
    # Expect a standard JSON list ([{}, {}]); fall back to JSONL if needed.
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Make sure the data is a list
            if not isinstance(data, list):
                print(f"Error: Expected a list of dictionaries in {json_file}")
                return {}, {}, pd.DataFrame()

            for doc in data:
                evaluated = evaluate_single_result(doc)
                results.append(evaluated)
                
    except json.JSONDecodeError:
        # If json.load fails, fall back to JSONL parsing.
        print("Warning: Standard JSON load failed, trying JSONL format...")
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    evaluated = evaluate_single_result(doc)
                    results.append(evaluated)
    
    print(f"Loaded {len(results)} predictions")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("No valid results to evaluate.")
        return {}, {}, results_df

    # Filter out OEA questions
    original_count = len(results_df)
    if 'skip' in results_df.columns:
        results_df = results_df[results_df['skip'] != True]
    skipped_count = original_count - len(results_df)
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} OEA/Unknown questions")
    
    # Aggregate results
    aggregated, detailed = aggregate_results(results_df)
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Print by category
    for category in ['Backward', 'Realtime', 'Forward']:
        print(f"\n{category}:")
        print("-" * 80)
        
        # Print detailed scores for this category
        if category in detailed:
            for key, value in sorted(detailed[category].items()):
                print(f"  {key:<60} {value*100:.2f}%")
        
        # Print average scores
        mca_key = f"{category}_MCA_avg"
        na_key = f"{category}_NA_avg"
        
        if na_key in aggregated:
            print(f"  {'[NA Average]':<60} {aggregated[na_key]*100:.2f}%")
        if mca_key in aggregated:
            print(f"\n  {'[MCA Average]':<60} {aggregated[mca_key]*100:.2f}%")
    
    # Print overall
    if 'Overall' in aggregated:
        print("\n" + "="*80)
        print(f"{'OVERALL SCORE:':<60} {aggregated['Overall']*100:.2f}%")
        print("="*80)
    
    return aggregated, detailed, results_df

if __name__ == "__main__":
    # Input file path
    input_file = "./output_logs/Stream3D-VLM-4B/evaluate_results.json"

    # Run evaluation
    aggregated_scores, detailed_scores, detailed_results = evaluate_predictions(input_file)


    output_scores = {
        "aggregated": {k: float(v) for k, v in aggregated_scores.items()},
        "detailed": {
            cat: {k: float(v) for k, v in scores.items()}
            for cat, scores in detailed_scores.items()
        }
    }