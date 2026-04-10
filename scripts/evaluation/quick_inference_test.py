from unsloth import FastLanguageModel
import torch

# 1. Load the Base Model + Your newly trained LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "nemotron_reasoning_lora_submission", # Points to your saved folder
    max_seq_length = 8192,
    dtype = None,
    load_in_4bit = True,
)

# Put Unsloth into ultra-fast inference mode
FastLanguageModel.for_inference(model) 

# 2. Format a Test Prompt (Pick a Knapsack problem it hasn't seen before!)
test_prompt = """You are an elite logical reasoning model. You must always think step-by-step inside <think>...</think> tags before providing your final answer. Your final answer must be strictly formatted and enclosed in \boxed{}.

Task ID: TEST-1
You are an expert logistics planner. You have a knapsack that can hold a maximum weight of 15kg.
You must choose a combination of the following items that maximizes the total value without exceeding the weight limit.

Available Items:
- Alpha: Weight = 5kg, Value = $50
- Beta: Weight = 10kg, Value = $60
- Gamma: Weight = 12kg, Value = $100

Format your final answer exactly like this: 'Items: [Item1, Item2] | Max Value: [Number]'.
Put your final formatted string inside \boxed{}."""

# Wrap it in ChatML exactly like the training data
messages = [
    {"role": "user", "content": test_prompt}
]

# Tokenize
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Tells the model it's the Assistant's turn to speak
    return_tensors = "pt",
).to("cuda")

# 3. Generate (Simulating the Kaggle Judge Parameters)
outputs = model.generate(
    input_ids = inputs,
    max_new_tokens = 7680, # Giving it the massive runway the judge allows
    temperature = 0.0,     # Greedy decoding (0.0)
    use_cache = True
)

# Decode and print the output!
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])