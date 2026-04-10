import json
import os

def create_balanced_eval_set(input_folder, output_file, rows_per_task=25):
    print(f"🧪 Creating 100-row Evaluation Set in {output_file}...")
    
    eval_records = []
    # Identify your 4 specific gold task types
    task_types = ['bitwise_gold.jsonl', 'knapsack_gold.jsonl', 
                  'logic_grid_gold.jsonl', 'stack_machine_gold.jsonl']
    
    # Ensure the validation directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for task_file in task_types:
        file_path = os.path.join(input_folder, task_file)
        
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: {task_file} not found. Skipping...")
            continue
            
        print(f"📥 Pulling {rows_per_task} rows from {task_file}...")
        
        count = 0
        with open(file_path, 'r') as f:
            for line in f:
                if count < rows_per_task:
                    eval_records.append(line.strip())
                    count += 1
                else:
                    break
        
        if count < rows_per_task:
            print(f"⚠️ Note: Only found {count} rows in {task_file}.")

    # Write the final balanced evaluation set
    with open(output_file, 'w') as f:
        for record in eval_records:
            f.write(record + '\n')

    print(f"✅ SUCCESS: Evaluation set with {len(eval_records)} rows created.")

if __name__ == "__main__":
    GOLD_DIR = "data/gold_standard"
    EVAL_FILE = "data/validation/eval_100.jsonl"
    create_balanced_eval_set(GOLD_DIR, EVAL_FILE)