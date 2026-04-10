import json
import os

def format_for_training(input_file, output_file):
    print(f"⚙️ Formatting {input_file} for GPU Fine-Tuning...")

    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found. Run the Rejection Sampler first!")
        return

    # The System Prompt Nemotron will actually use during inference after training
    # This teaches the model its identity and its core mechanical constraint
    SYSTEM_PROMPT = (
        "You are an elite logical reasoning model. You must always think step-by-step "
        "inside <think>...</think> tags before providing your final answer. "
        "Your final answer must be strictly formatted and enclosed in \\boxed{}."
    )

    formatted_count = 0

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            try:
                task = json.loads(line)
                
                # Extract the pure prompt and perfect response
                user_prompt = task.get("prompt", "")
                assistant_response = task.get("teacher_response", "")

                # Skip any malformed rows just in case
                if not user_prompt or not assistant_response:
                    continue

                # Build the exact ChatML / Messages array expected by Hugging Face & Unsloth
                chatml_format = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_response}
                    ]
                }

                # Write the formatted object to our final training file
                outfile.write(json.dumps(chatml_format) + '\n')
                formatted_count += 1

            except json.JSONDecodeError:
                continue

    print("==========================================")
    print("🚀 FORMATTING COMPLETE")
    print("==========================================")
    print(f"✅ Successfully formatted {formatted_count} perfect examples.")
    print(f"💾 Ready for GPU upload: {output_file}")

if __name__ == "__main__":
    # Point this at the output from Phase 3 (your 100% accurate filtered data)
    INPUT_JSONL = "nemotron_training_gold_knapsack.jsonl"
    OUTPUT_JSONL = "nemotron_chatml_knapsack.jsonl"
    
    format_for_training(INPUT_JSONL, OUTPUT_JSONL)