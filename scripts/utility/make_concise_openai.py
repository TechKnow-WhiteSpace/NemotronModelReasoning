import json
import time
import os
from openai import OpenAI

# 1. Safely pull the API key from the environment
# Checking both your custom OPEN_API_KEY and the standard OPENAI_API_KEY
api_key = os.environ.get("OPEN_API_KEY") or os.environ.get("OPENAI_API_KEY")

# Safety check to prevent crashing mid-run if the key is missing
if not api_key:
    raise ValueError("❌ OPEN_API_KEY environment variable is not set! Please export it in your terminal before running.")

# Initialize the OpenAI Client
client = OpenAI(api_key=api_key)

input_file = 'data/gold_standard/nemotron_training_gold_master.jsonl'
output_file = 'data/gold_standard/nemotron_training_gold_v6.jsonl'

compression_system_prompt = """
[ROLE]
You are an Expert Data Distillation Engineer specializing in LLM test-time compute optimization.

[CONTEXT]
You are translating verbose Chain-of-Thought (CoT) reasoning into "Structured Verbosity." We must remove conversational fluff to optimize context limits, but we MUST preserve the dense, step-by-step logic required for a reasoning model to successfully "think" through complex puzzles.

[CONSTRAINTS]
1. STRIP all conversational filler, pleasantries, and redundant explanations (e.g., "Let's solve this step-by-step," "We can clearly see that," "Now we will calculate").
2. RETAIN every single mathematical equation, variable assignment, state change, and deductive logical branch. DO NOT skip, abstract, or abbreviate the actual work.
3. FORMAT the output as highly dense, structured sequential logic. Use rigorous bullet points, mathematical notation, or logic-gate style statements.
4. DO NOT over-compress. The model needs a sufficient number of reasoning tokens (test-time compute) to reach the correct answer. 

[OUTPUT]
Output ONLY the distilled, step-by-step reasoning logic. Do not include introductory or concluding conversational remarks.
"""

def extract_think_block(completion_text):
    # Extracts text between <think> and </think>
    start = completion_text.find('<think>') + 7
    end = completion_text.find('</think>')
    if start == 6 or end == -1:
        return None, completion_text
    return completion_text[start:end].strip(), completion_text[end+8:].strip()

print("🚀 Starting Secure OpenAI CoT Compression Pipeline...")

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line_num, line in enumerate(infile):
        data = json.loads(line)
        
        # Look for 'teacher_response' first, fallback to 'completion'
        original_completion = data.get('teacher_response') or data.get('completion')
        
        if not original_completion:
            print(f"⚠️ Warning on row {line_num + 1}: Could not find text. Skipping.")
            outfile.write(json.dumps(data) + '\n')
            continue
            
        # Split the reasoning from the final \boxed{} answer
        think_text, answer_text = extract_think_block(original_completion)
        
        if think_text:
            try:
                # Ask gpt-4o-mini to compress the reasoning
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": compression_system_prompt},
                        {"role": "user", "content": f"ORIGINAL TEXT:\n{think_text}"}
                    ],
                    temperature=0.2 # Low temperature for strict analytical formatting
                )
                
                compressed_think = response.choices[0].message.content.strip()
                
                # Stitch it back together
                new_completion = f"<think>\n{compressed_think}\n</think>\n\n{answer_text}"
                
                # Save it back to the exact key it came from
                if 'teacher_response' in data:
                    data['teacher_response'] = new_completion
                else:
                    data['completion'] = new_completion
                
                # Write the new compressed row to our V5 file
                outfile.write(json.dumps(data) + '\n')
                print(f"✅ Processed row {line_num + 1} (Tokens reduced significantly)")
                
                # Brief sleep to avoid hitting OpenAI rate limits on lower-tier accounts
                time.sleep(0.5) 
                
            except Exception as e:
                print(f"⚠️ Error on row {line_num + 1}: {e}. Skipping compression for this row.")
                outfile.write(json.dumps(data) + '\n')
        else:
            # If no <think> tags found, just write it as-is
            outfile.write(json.dumps(data) + '\n')

print(f"🎉 Pipeline complete! Compressed dataset saved to {output_file}")