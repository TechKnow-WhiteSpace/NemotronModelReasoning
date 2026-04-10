import pandas as pd
import json

def convert_kaggle_train_to_jsonl(csv_path, output_jsonl):
    print(f"🔄 Converting {csv_path} to Pipeline Format...")
    
    # Load the Kaggle training data
    df = pd.read_csv(csv_path)
    
    converted_count = 0
    with open(output_jsonl, 'w') as f_out:
        # Iterate through the Kaggle rows
        for index, row in df.iterrows():
            # Extract using the exact column names from your CSV
            task_id = str(row['id'])
            question_text = str(row['prompt'])
            answer_text = str(row['answer']).strip() 
            
            # 1. The Prompt we will send to the Teacher API
            prompt = (
                f"Task ID: {task_id}\n"
                f"Solve the following puzzle step-by-step.\n\n"
                f"Puzzle:\n{question_text}\n\n"
                f"Use <think> tags to formulate your logic. "
                f"Format your final answer exactly and put it inside \\boxed{{}}."
            )
            
            # 2. The Ground Truth for our Rejection Sampler to grade against
            ground_truth = {
                "domain": "Kaggle_Bitwise_Baseline",
                "original_id": task_id, # Using Kaggle's exact ID for tracking
                "boxed_answer": answer_text 
            }
            
            # Build the unified object
            task_obj = {
                "prompt": prompt,
                "ground_truth": ground_truth
            }
            
            f_out.write(json.dumps(task_obj) + '\n')
            converted_count += 1
            
    print(f"✅ Successfully formatted {converted_count} Kaggle tasks for the Pipeline.")
    print(f"📁 Saved to {output_jsonl}")

if __name__ == "__main__":
    # Ensure this points to your Kaggle file
    KAGGLE_CSV = "../../data/seed_files/train.csv" 
    OUTPUT_FILE = "../../data/seed_files/bitwise_dataset.jsonl"
    
    convert_kaggle_train_to_jsonl(KAGGLE_CSV, OUTPUT_FILE)