import json
import os

# 1. DEFINE YOUR SOURCES
data_map = {
    "bitwise": "data/gold_standard/bitwise/nemotron_training_gold_bitwise_master.jsonl",
    "knapsack": "data/gold_standard/knapsack/nemotron_training_gold_knapsack.jsonl",
    "logic_grid": "data/gold_standard/logic_grid/nemotron_training_gold_logic_grid.jsonl",
    "stack_machine": "data/gold_standard/stack_machine/nemotron_training_gold_stack_machine.jsonl"
}

output_file = "data/gold_standard/nemotron_training_gold_master.jsonl"
mega_data = []

print("📂 Starting Mega-Merge...")

for domain, path in data_map.items():
    if not os.path.exists(path):
        print(f"⚠️ Warning: {path} not found. Skipping {domain}.")
        continue
        
    count = 0
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            
            # UNIFY STRUCTURE: Ensure every row has 'prompt' and 'completion'
            # Fallback to 'teacher_response' if 'completion' isn't there yet
            prompt = item.get("prompt")
            completion = item.get("completion") or item.get("teacher_response")
            
            if prompt and completion:
                mega_data.append({
                    "prompt": prompt,
                    "completion": completion,
                    "domain": domain # Useful for tracking performance per category
                })
                count += 1
    print(f"✅ Loaded {count} rows from {domain}.")

# 2. SHUFFLE AND SAVE
# Shuffling is CRITICAL so the model doesn't "forget" bitwise while learning knapsack
import random
random.shuffle(mega_data)

with open(output_file, 'w') as f_out:
    for entry in mega_data:
        f_out.write(json.dumps(entry) + '\n')

print(f"🎉 SUCCESS: {len(mega_data)} rows merged and shuffled into {output_file}")