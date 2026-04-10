import re
import json
import time
import os
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

# ==========================================
# 1. API SETUP
# ==========================================
# Assumes GEMINI_API_KEY is exported in your environment variables
if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("⚠️ GEMINI_API_KEY not found in environment variables.")

client = genai.Client()

# Using Gemini 2.5 Pro as our elite reasoning Teacher
MODEL_NAME = "gemini-2.5-pro"

# ==========================================
# 2. THE RCCO TEACHER PROMPT
# ==========================================
TEACHER_SYSTEM_PROMPT = """
You are an elite logical reasoning engine. Your purpose is to solve complex puzzles with absolute mathematical and deductive perfection.
You MUST follow these strict constraints:
1. Use <think>...</think> tags for all step-by-step logic, hypotheses, and verification.
2. Your final answer MUST be strictly formatted according to the prompt's requirements.
3. You MUST wrap your final formatted string inside \\boxed{}.
4. **CRITICAL**: The \\boxed{} tag MUST be the very last line of your response. Do not include any text, signatures, or filler after the box.
5. **VERIFICATION**: Before finishing, mentally check:
   - Is my reasoning inside <think> tags?
   - Does my final string exactly match the required output format?
   - Is my final answer inside \\boxed{}?
6. Example end-of-response:
   ...and therefore the decryption is complete.
   \\boxed{the clever dragon creates}
"""

# ==========================================
# 3. EXPONENTIAL BACKOFF WRAPPER
# ==========================================
# If we hit an API limit, wait 4s, then 8s, then 16s, up to 5 times.
#@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=30))
def generate_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=TEACHER_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=65535,
                    # --- FIXED: Correct 2026 SDK Category and Threshold strings ---
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH", 
                            threshold="BLOCK_NONE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT", 
                            threshold="BLOCK_NONE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_SEXUALLY_EXPLICIT", 
                            threshold="BLOCK_NONE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT", 
                            threshold="BLOCK_NONE"
                        ),
                    ],
                    http_options={'timeout': 300000} 
                )
            )
            
            # Check if we actually got a candidate with text
            if response.candidates and response.candidates[0].content.parts:
                return response.text
            
            # If no text, check the finish reason for debugging
            finish_reason = response.candidates[0].finish_reason if response.candidates else "Unknown"
            print(f"⚠️ Attempt {attempt+1}: No text. Finish Reason: {finish_reason}")
            
        except Exception as e:
            print(f"❌ Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** (attempt + 1))
            
    return None

# ==========================================
# 4. THE ENHANCED EXECUTION PIPELINE
# ==========================================
import re

def extract_id_from_prompt(prompt_text):
    """Helper to pull '00066667' out of the 'Task ID: 00066667' string."""
    match = re.search(r"Task ID:\s*(\S+)", str(prompt_text))
    return match.group(1) if match else None

def process_dataset(input_filename, output_filename, failure_log="data/synthetic_factory/bitwise/failures.jsonl", max_new_successes=500):
    print(f"🚀 Initializing Symmetric ID Pipeline...")
    
    # 1. Load Solved IDs by parsing their prompt strings
    solved_ids = set()
    if os.path.exists(output_filename):
        with open(output_filename, 'r') as f:
            for line in f:
                item = json.loads(line)
                # Look for root ID first (for new data) then fallback to prompt string (for old data)
                tid = item.get("id") or extract_id_from_prompt(item.get("prompt", ""))
                if tid: solved_ids.add(str(tid))
    
    # 2. Load Failed IDs (to avoid re-trying toxic puzzles)
    failed_ids = set()
    if os.path.exists(failure_log):
        with open(failure_log, 'r') as f:
            for line in f:
                fid = json.loads(line).get("id")
                if fid: failed_ids.add(str(fid))

    with open(input_filename, 'r') as f:
        tasks = [json.loads(line) for line in f]

    new_success_count = 0
    print(f"📂 Solved: {len(solved_ids)} | Blacklisted: {len(failed_ids)} | Total Tasks: {len(tasks)}")

    with open(output_filename, 'a') as f_out, open(failure_log, 'a') as f_fail:
        for task in tasks:
            # EXTRACT ID FROM THE CURRENT SEED TASK
            task_id = extract_id_from_prompt(task.get("prompt", ""))
            
            if not task_id:
                continue # Skip if no ID found
                
            if task_id in solved_ids or task_id in failed_ids:
                continue # SKIP confirmed!
                
            if new_success_count >= max_new_successes:
                print(f"\n🛑 Target reached ({max_new_successes}). Stopping.")
                break

            print(f"🧠 Task {task_id}...", end=" ", flush=True)
            raw_cot_output = generate_with_retry(task["prompt"])
            
            if raw_cot_output:
                task["id"] = task_id  # Inject the ID into the record for future easy loading
                task["teacher_response"] = raw_cot_output
                f_out.write(json.dumps(task) + '\n')
                new_success_count += 1
                print(f"✅ Success ({new_success_count}/{max_new_successes})")
                time.sleep(2)
            else:
                print("❌ Failed. Blacklisting.")
                f_fail.write(json.dumps({"id": task_id, "status": "failed_after_retries"}) + '\n')
                time.sleep(5)

if __name__ == "__main__":
    # Ensure these point to the files you just generated!
    INPUT_FILE = "data/seed_files/bitwise_dataset.jsonl" 
    OUTPUT_FILE = "data/synthetic_factory/bitwise/bitwise_cot_dataset.jsonl"
    
    process_dataset(INPUT_FILE, OUTPUT_FILE)