
import json
import re
import os

def extract_boxed_answer(text):
    """Finds the LAST \boxed{} in the teacher response."""
    if not text: return None
    # Matches everything inside the last \boxed{...}
    matches = re.findall(r"\\boxed\{(.+?)\}", str(text))
    return matches[-1].strip() if matches else None

def normalize_answer(val):
    """
    The 'Fuzzy Matcher': Strips all descriptive text and punctuation.
    Converts 'Alice lives in the Blue house' -> 'alicebluehouse'
    """
    if val is None: return ""
    s = str(val).lower()
    
    # 1. Remove common descriptive labels across all domains
    labels = [
        'items', 'max value', 'final top value', 'final stack', 
        'lives', 'in', 'the', 'house', 'and', 'owns', 'correct', 
        'value', 'person', 'color', 'pet', 'total', 'selection'
    ]
    for label in labels:
        s = s.replace(label + ":", "").replace(label, "")
    
    # 2. Remove all non-alphanumeric characters
    s = re.sub(r'[^a-z0-9]', ' ', s)
    
    # 3. Split into words, sort them, and join
    # Sorting handles 'Alpha, Beta' matching 'Beta, Alpha'
    parts = sorted([p for p in s.split() if p])
    return "".join(parts)

def process_domain(domain_name, input_file, output_file):
    print(f"🕵️‍♂️ Processing {domain_name}...")
    
    total = 0
    perfect = 0
    failed = 0
    missing_box = 0
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            total += 1
            item = json.loads(line)
            
            ground_truth = item.get("ground_truth", {}).get("boxed_answer", "")
            model_raw = extract_boxed_answer(item.get("teacher_response", ""))
            
            if not model_raw:
                missing_box += 1
                continue
                
            # Perform the aggressive normalized comparison
            if normalize_answer(model_raw) == normalize_answer(ground_truth):
                perfect += 1
                f_out.write(json.dumps(item) + '\n')
            else:
                # OPTIONAL: Print first few failures for debugging
                if failed < 3:
                    print(f"  ❌ Debug ({domain_name}):")
                    print(f"     Model: {model_raw} -> {normalize_answer(model_raw)}")
                    print(f"     Truth: {ground_truth} -> {normalize_answer(ground_truth)}")
                failed += 1

    yield_rate = (perfect / total) * 100 if total > 0 else 0
    print(f"✅ {domain_name} Complete | Yield: {yield_rate:.1f}% | Perfect: {perfect}/{total}\n")

if __name__ == "__main__":
    # Define the tasks to process
    tasks = [
        ("Knapsack", 
         "data/synthetic_factory/knapsack/knapsack_cot_dataset.jsonl", 
         "data/gold_standard/knapsack/nemotron_training_gold_knapsack.jsonl"),
        
        ("Stack Machine", 
         "data/synthetic_factory/stack_machine/stack_machine_cot_dataset.jsonl", 
         "data/gold_standard/stack_machine/nemotron_training_gold_stack_machine.jsonl"),
        
        ("Logic Grid", 
         "data/synthetic_factory/logic_grid/logic_grid_cot_dataset.jsonl", 
         "data/gold_standard/logic_grid/nemotron_training_gold_logic_grid.jsonl")
    ]
    
    print("==========================================")
    print("🎯 MULTI-DOMAIN REJECTION SAMPLER START")
    print("==========================================\n")
    
    for domain, fin, fout in tasks:
        if os.path.exists(fin):
            process_domain(domain, fin, fout)
        else:
            print(f"⚠️ Skipping {domain}: File not found.\n")