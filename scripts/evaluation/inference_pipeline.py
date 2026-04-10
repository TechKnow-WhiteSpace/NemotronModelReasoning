import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import kagglehub

# --- STEP 1: LOAD THE TEST DATA ---
print("📊 Loading test.csv...")
test_df = pd.read_csv('/kaggle/input/[YOUR-COMPETITION-NAME]/test.csv')
# Create an empty list to store our generated answers
predictions = []

# --- STEP 2: LOAD THE PATCHED BASE MODEL ---
# (We must use the exact same bnb_config we trained with, including the lm_head bypass!)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_skip_modules=["in_proj", "out_proj", "x_proj", "dt_proj", "conv1d", "lm_head"]
)

print("📦 Loading Base Model...")
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")

# NOTE: If the Kaggle submission environment doesn't allow internet, 
# you will load the base model from a Kaggle Dataset instead of kagglehub!
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# --- STEP 3: ATTACH YOUR TRAINED LORA ADAPTER ---
print("🧠 Attaching your custom RCCO-trained adapter...")
# Point this to wherever you uploaded your submission.zip/final_adapter folder
ADAPTER_PATH = "/kaggle/input/[YOUR-UPLOADED-ADAPTER-DATASET]/final_adapter"
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# --- STEP 4: GENERATE PREDICTIONS ---
print("🚀 Starting Inference Loop...")
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        
        # Loop through every row in the test.csv
        for index, row in test_df.iterrows():
            print(f"Processing ID: {row['id']}...")
            
            # 1. Format the prompt EXACTLY how we trained it
            formatted_prompt = f"{row['prompt']}\n\n### Reasoning:\n"
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
            
            # 2. Generate the answer
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,       # Give it enough room to explain its reasoning
                temperature=0.7,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # 3. Decode the output
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 4. Clean up: We only want the *new* text, not the prompt echoed back
            # We split the string at our specific keyword and take everything after it
            generated_answer = full_response.split("### Reasoning:\n")[-1].strip()
            
            predictions.append(generated_answer)

# --- STEP 5: SAVE THE SUBMISSION ---
print("💾 Saving submission.csv...")
# Overwrite the prompt column (or create a new 'response' column depending on comp rules)
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'response': predictions # Check the competition rules to see what they want this column named!
})

submission_df.to_csv('submission.csv', index=False)
print("✅ Done! Ready to submit to the leaderboard.")