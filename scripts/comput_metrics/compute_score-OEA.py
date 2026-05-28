import os
import json
import re
import argparse
import concurrent.futures
import random
from tqdm import tqdm
from typing import List, Dict

# OpenAI dependencies
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)

# ============== Question type definitions ==============
Backward = [
    "camera_direction_backward", "camera_rotation_backward",
    "object2camera_direction_backward", "object_counting_backward",
    "appearance_order_backward", "camera_displacement_backward",
    "camera_path_backward", "object2camera_distance_backward",
    "appearance_time_backward", "camera_comprehensive_backward",
    "object2camera_location_backward", "room_area_backward"
]

Realtime = [
    "object2camera_reldistance_realtime", "object2object_distance_realtime",
    "object2camera_distance_realtime", "object_size_realtime",
    "object_property_realtime", "object_position_realtime",
    "object_recognition_realtime"   
]

Forward = [
    "camera_direction_forward", "camera_rotation_forward",
    "object2camera_direction_forward", "camera_displacement_forward",
    "camera_path_forward", "object2camera_distance_forward",
    "camera_comprehensive_forward", "object2camera_location_forward",
    "object_finding_forward", "room_area_forward"
]

# OEA question types to evaluate
OEA_QUESTION_TYPES = [
    "camera_comprehensive_backward", "camera_comprehensive_forward",
    "object2camera_location_backward", "object2camera_location_forward",
    "room_area_backward", "room_area_forward",
    "object_finding_forward", "object_size_realtime",
    "object_property_realtime", "object_position_realtime",
    "object_recognition_realtime"
]

# ============== API configuration ==============
client = OpenAI(
    api_key='',
    base_url=''
)

# ============== Utility functions ==============
def get_category(question_type):
    if question_type in Backward: return 'Backward'
    if question_type in Realtime: return 'Realtime'
    if question_type in Forward: return 'Forward'
    return 'Unknown'

def load_template(template_path):
    if not os.path.exists(template_path):
        return """
You are an intelligent assistant evaluating the correctness of a model's prediction against a ground truth answer.

Question: {{{question}}}
Ground Truth: {{{ground_truth}}}
Prediction: {{{prediction}}}

Compare the Prediction with the Ground Truth.
Adopt a lenient evaluation standard:
- If the Prediction mentions the key keywords, entities, or concepts found in the Ground Truth, output {"verdict": "Correct"}.
- If the Prediction is broadly correct or represents a reasonable interpretation of the question, output {"verdict": "Correct"}.
- Do not penalize for missing details or slightly different interpretations.

Output {"verdict": "Incorrect"} only if the Prediction is clearly wrong or completely unrelated to the Ground Truth.

Provide your response in JSON format.
"""
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()

def render_prompt(template, **kwargs):
    for key, value in kwargs.items():
        template = template.replace(r"{{{" + key + r"}}}", str(value))
    return template


def parse_verdict_from_response(response):
    response_lower = response.lower().strip()
    
    # 1. Try to parse as JSON
    try:
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            verdict = data.get("verdict")
            if str(verdict).lower() in ["correct", "true", "yes", "1"]: return True
            if str(verdict).lower() in ["incorrect", "false", "no", "0"]: return False
    except:
        pass
    
    # 2. Keyword matching
    if '"verdict": "correct"' in response_lower: return True
    if '"verdict": "incorrect"' in response_lower: return False
    if "correct" in response_lower and "incorrect" not in response_lower: return True
    if "incorrect" in response_lower: return False
    
    return None


# ============== API call logic ==============

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception)
)
def call_gpt4o_api(messages: List[Dict]) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=200,
        temperature=0,
        stream=True
    )
    full_content = ""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            full_content += chunk.choices[0].delta.content
    return full_content

def evaluate_single_sample(sample, template, test_mode=False):
    """Evaluate a single sample. Returns (is_oea, is_correct, updated_sample)."""
    question_type = sample.get("question_type", "Unknown")

    # 1. Skip non-OEA samples
    if question_type not in OEA_QUESTION_TYPES:
        return False, None, sample

    # 2. Prepare data
    question = sample.get("query", "")
    gt = sample.get("ground_truth", "").split("<image>")[-1].strip()
    pd = sample.get("prediction", "").split("<image>")[-1].strip()

    # 3. Build prompt
    prompt_text = render_prompt(template, question=question, ground_truth=gt, prediction=pd)
    messages = [
        {"role": "system", "content": "You are an expert evaluator. Output ONLY a JSON object with a 'verdict' field."},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
    ]

    # 4. Call or mock the API
    if test_mode:
        is_correct = random.choice([True, False])
        response_text = "Mock Response"
    else:
        try:
            response_text = call_gpt4o_api(messages)
            is_correct = parse_verdict_from_response(response_text)
        except Exception as e:
            print(f"API Error: {e}")
            is_correct = None
            response_text = str(e)

    # 5. Update sample info
    sample['extracted_prediction'] = pd
    sample['gpt4o_response'] = response_text
    sample['eval'] = "Correct" if is_correct else ("Incorrect" if is_correct is False else "Failed")
    sample['category'] = get_category(question_type)
    
    return True, is_correct, sample

# ============== Main ==============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="")
    parser.add_argument('--template_path', type=str, default="promp4eval.txt")
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--max_workers', type=int, default=10)
    args = parser.parse_args()
    
    # Load data
    print(f"📂 Loading: {args.input_file}")
    samples = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        # Try parsing as a JSON array
        if content.startswith('['):
            try:
                samples = json.loads(content)
                print(f"✅ Loaded as JSON array: {len(samples)} samples")
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse as JSON array: {e}")
                raise
        else:
            # Otherwise parse as JSONL, line by line
            for line in content.split('\n'):
                if line.strip():
                    samples.append(json.loads(line))
            print(f"✅ Loaded as JSONL: {len(samples)} samples")
            
    if args.num_samples: samples = samples[:args.num_samples]
    template = load_template(args.template_path)
    
    print(f"🚀 Starting Evaluation (OEA Only, Workers={args.max_workers})...")
    
    # Parallel evaluation
    evaluated_results = []
    stats = {
        'Backward': {'correct': 0, 'total': 0},
        'Realtime': {'correct': 0, 'total': 0},
        'Forward':  {'correct': 0, 'total': 0},
        'Unknown':  {'correct': 0, 'total': 0}
    }
    
    skipped_count = 0
    failed_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(evaluate_single_sample, s, template, args.test_mode): s for s in samples}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(samples)):
            try:
                is_oea, is_correct, sample = future.result()
                evaluated_results.append(sample)
                
                if not is_oea:
                    skipped_count += 1
                    continue
                
                cat = sample['category']
                if cat not in stats: cat = 'Unknown'
                
                if is_correct is not None:
                    stats[cat]['total'] += 1
                    if is_correct: stats[cat]['correct'] += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                print(f"Worker Error: {e}")
                failed_count += 1

    # Report results
    print(f"\n{'='*60}")
    print(f"📊 OEA Scores by Category")
    print(f"{'='*60}")
    
    total_correct = 0
    total_evaluated = 0
    
    for cat in ['Backward', 'Realtime', 'Forward']:
        s = stats[cat]
        acc = (s['correct'] / s['total'] * 100) if s['total'] > 0 else 0.0
        print(f"{cat:<10} | Correct: {s['correct']:<4} | Total: {s['total']:<4} | Acc: {acc:.2f}%")
        total_correct += s['correct']
        total_evaluated += s['total']
        
    total_acc = (total_correct / total_evaluated * 100) if total_evaluated > 0 else 0.0
    
    print(f"{'-'*60}")
    print(f"Overall OEA | Correct: {total_correct:<4} | Total: {total_evaluated:<4} | Acc: {total_acc:.2f}%")
    print(f"Skipped (Non-OEA): {skipped_count}")
    print(f"Failed (API/Parse): {failed_count}")
    print(f"{'='*60}\n")
    
    # Save results
    sample_suffix = f"_{args.num_samples}samples" if args.num_samples else "_all"
    output_path = os.path.join(os.path.dirname(args.input_file) or ".", f"evaluation_results_oea{sample_suffix}.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": stats, 
            "overall_accuracy": total_acc,
            "samples": evaluated_results
        }, f, indent=2, ensure_ascii=False)
    print(f"💾 Results saved to: {output_path}")

if __name__ == "__main__":
    main()