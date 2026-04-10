import os

def consolidate_gold_data(file_list, master_output):
    total_rows = 0
    with open(master_output, 'w') as master_f:
        for file_path in file_list:
            if os.path.exists(file_path):
                print(f"📥 Merging {file_path}...")
                with open(file_path, 'r') as f:
                    for line in f:
                        master_f.write(line)
                        total_rows += 1
            else:
                print(f"⚠️ Warning: {file_path} not found.")
    
    print(f"✅ SUCCESS: {total_rows} Golden rows consolidated into {master_output}")

# Your current progress files
GOLD_FILES = [
    "data/gold_standard/bitwise/nemotron_training_gold_bitwise.jsonl",
    "data/gold_standard/bitwise/nemotron_training_gold_bitwise_50.jsonl"
]
MASTER_GOLD = "data/gold_standard/bitwise/nemotron_training_gold_bitwise_master.jsonl"

consolidate_gold_data(GOLD_FILES, MASTER_GOLD)