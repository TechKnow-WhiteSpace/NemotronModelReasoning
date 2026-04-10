import json
import time
from google import genai
import os

# 1. Safely pull the API key from the environment
api_key = os.environ.get("GEMINI_API_KEY")

# Safety check to prevent crashing mid-run if the key is missing
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY environment variable is not set! Please export it in your terminal before running.")

# Initialize the modern GenAI Client securely
client = genai.Client(api_key=api_key)

input_file = 'data/gold_standard/bitwise/nemotron_training_gold_bitwise_master.jsonl'
output_file = 'data/gold_standard/bitwise/nemotron_training_v5_bitwise.jsonl'

compression_prompt = """
Compress this reasoning into ultra-concise shorthand pseudo-code. 
Strip all filler words. Preserve the exact math and logic steps.
Output ONLY the compressed text. Maximum 150 words.
"""

def extract_think_block(completion_text):
    # Extracts text between <think> and </think>
    start = completion_text.find('<think>') + 7
    end = completion_text.find('</think>')
    if start == 6 or end == -1:
        return None, completion_text
    return completion_text[start:end].strip(), completion_text[end+8:].strip()

print("🚀 Starting CoT Compression Pipeline...")

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line_num, line in enumerate(infile):
        data = json.loads(line)
        
        # FIX: Look for 'teacher_response' first, fallback to 'completion'
        original_completion = data.get('teacher_response') or data.get('completion')
        
        if not original_completion:
            print(f"⚠️ Warning on row {line_num + 1}: Could not find text. Skipping.")
            outfile.write(json.dumps(data) + '\n')
            continue
            
        # 1. Split the reasoning from the final \boxed{} answer
        think_text, answer_text = extract_think_block(original_completion)
        
        if think_text:
            try:
                # 2. Ask the API to compress the reasoning
                response = client.models.generate_content(
                    model='gemini-2.5-pro',
                    contents=f"{compression_prompt}\n\nORIGINAL TEXT:\n{think_text}"
                )
                compressed_think = response.text.strip()
                
                # 3. Stitch it back together
                new_completion = f"<think>\n{compressed_think}\n</think>\n\n{answer_text}"
                
                # FIX: Save it back to the exact key it came from!
                if 'teacher_response' in data:
                    data['teacher_response'] = new_completion
                else:
                    data['completion'] = new_completion
                
                # Write the new compressed row to our V5 file
                outfile.write(json.dumps(data) + '\n')
                print(f"✅ Processed row {line_num + 1} (Tokens reduced significantly)")
                
                # Brief sleep to respect free-tier rate limits
                time.sleep(1) 
                
            except Exception as e:
                print(f"⚠️ Error on row {line_num + 1}: {e}. Skipping compression for this row.")
                outfile.write(json.dumps(data) + '\n')
        else:
            # If no <think> tags found, just write it as-is
            outfile.write(json.dumps(data) + '\n')

print(f"🎉 Pipeline complete! Compressed dataset saved to {output_file}")