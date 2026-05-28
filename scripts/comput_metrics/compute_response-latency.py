import json

def calc_avg_from_file(file_path):
    metrics = ["ttft_seconds", "ttft_ms", "answer_generation_latency_seconds", "answer_generation_latency_ms"]
    totals = {k: 0.0 for k in metrics}
    count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Try as a standard JSON list first.
        try:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if "timing" in item:
                        for k in metrics:
                            totals[k] += item["timing"].get(k, 0)
                        count += 1
        except json.JSONDecodeError:
            # Fall back to JSONL (one JSON per line).
            f.seek(0)
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if "timing" in item:
                        for k in metrics:
                            totals[k] += item["timing"].get(k, 0)
                        count += 1

    if count > 0:
        print(f"Total samples: {count}")
        for k in metrics:
            print(f"Avg {k}: {totals[k]/count}")

input_file = ""
calc_avg_from_file(input_file)