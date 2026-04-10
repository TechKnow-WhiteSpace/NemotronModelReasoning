import os
import pandas as pd
import json
import time
from google import genai
from google.genai import types

# ==========================================
# 1. SETUP & AUTHENTICATION
# ==========================================
# Assuming you still have your GEMINI_API_KEY exported in your terminal
if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("⚠️ GEMINI_API_KEY not found in environment variables.")

client = genai.Client()
# We want a powerful reasoning model for the Teacher. Gemini 2.5 Pro is ideal here.
teacher_model = 'gemini-2.5-pro'

# File paths
INPUT_CSV = 'data/train.csv'
OUTPUT_JSONL = 'data/synthetic_reasoning_dataset.jsonl'

# ==========================================
# 2. THE RCCO TEACHER PROMPT
# ==========================================
RCCO_TEACHER_PROMPT = """
**Role:**
You are an elite Cryptographer and Logic Mathematician. Your specialty is reverse-engineering hidden transformation rules, complex algebra, and logic puzzles.

**Context:**
I am building a synthetic reasoning dataset to train a 30B parameter model. 
I will provide you with a [Puzzle Prompt] and the [Final Correct Answer]. 
Your job is to generate the flawless, step-by-step logical deduction required to reach that exact answer.

[Puzzle Prompt]: {question}
[Final Correct Answer]: {answer}

**Constraints:**
1. You MUST explicitly state your hypotheses for what the hidden rule or logical path is.
2. You MUST test your hypothesis against the provided examples (if any) to prove it works.
3. Your logic MUST perfectly arrive at the provided [Final Correct Answer].
4. You MUST enclose all of your scratchpad reasoning inside <think> tags.
5. You MUST output the exact final answer inside a \\boxed{{}} tag at the very end.

**Output Format:**
Return ONLY the reasoning block and the final boxed answer. Do not include markdown code blocks.
<think>
[Your step-by-step logical deduction]
</think>
\\boxed{{{answer}}}
"""

# ==========================================
# 3. PIPELINE EXECUTION
# ==========================================
def run_pipeline():
    print(f"📂 Loading data from {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"❌ Could not find {INPUT_CSV}. Please download it from Kaggle.")
        return

    # For testing, let's just do the first 5 rows so we don't burn your API quota blindly
    test_df = df.head(5) 
    print(f"🚀 Generating synthetic Chain-of-Thought for {len(test_df)} puzzles...")

    successful_generations = []

    for index, row in test_df.iterrows():
        question = row.get('Question', row.get('prompt', '')) # Adjust column name based on CSV
        answer = row.get('Response', row.get('answer', ''))   # Adjust column name based on CSV
        
        formatted_prompt = RCCO_TEACHER_PROMPT.format(question=question, answer=answer)
        
        print(f"   ⏳ Processing Puzzle #{index + 1}...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=teacher_model,
                    contents=formatted_prompt,
                    # Lower temperature means highly logical, deterministic output
                    config=types.GenerateContentConfig(temperature=0.1) 
                )
                
                # Format exactly how standard LLM fine-tuning scripts expect it
                training_example = {
                    "text": f"User: {question}\n\nAssistant: {response.text}"
                }
                successful_generations.append(training_example)
                
                # Be nice to the API
                time.sleep(3) 
                break
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"   ❌ Failed on Puzzle #{index + 1}: {e}")
                else:
                    print(f"   ⚠️ Rate limit/Error. Retrying in 10s... ({e})")
                    time.sleep(10)

    # ==========================================
    # 4. SAVE TO JSONL
    # ==========================================
    with open(OUTPUT_JSONL, 'w') as f:
        for item in successful_generations:
            f.write(json.dumps(item) + '\n')
            
    print(f"🎉 Successfully generated {len(successful_generations)} training examples!")
    print(f"💾 Saved to {OUTPUT_JSONL}")

if __name__ == "__main__":
    run_pipeline()