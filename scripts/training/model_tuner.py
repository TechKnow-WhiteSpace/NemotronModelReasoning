import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ==========================================
# 1. HARDWARE & MODEL CONFIGURATION
# ==========================================
# We need a large sequence length because our <think> tags generate massive walls of text
max_seq_length = 8192 # <-- BUMPED up to match the evaluation max_model_len 
dtype = None # Auto-detects bf16 for Blackwell GPUs
load_in_4bit = True # CRITICAL: Shrinks the 30B model to fit on the 48GB RTX 6000

print("🚀 Loading Nemotron-3-Nano-30B via Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "nvidia/Nemotron-3-Nano-30B", 
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_your_token_here", # Uncomment if Nemotron is gated on Hugging Face
)

# ==========================================
# 2. INJECTING THE LORA ADAPTERS
# ==========================================
print("⚙️ Attaching LoRA Adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # <-- BUMPED to the max_lora_rank limit
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32, # <-- Keep alpha matched with rank
    lora_dropout = 0, 
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Unsloth's secret weapon for saving VRAM
    random_state = 3407,
)

# ==========================================
# 3. LOADING THE BALANCED SFT DATA
# ==========================================
print("📚 Loading Shuffled, Leakage-Proof Reasoning Dataset...")
# Point this to the output of our Final Merge & Shuffle script!
dataset = load_dataset("json", data_files="data/gold_standard/bitwise/nemotron_training_gold_bitwise.jsonl", split="train")


# Unsloth has a built-in ChatML formatter that maps our system/user/assistant roles to the model
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml", 
    mapping = {"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"}
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts }

dataset = dataset.map(formatting_prompts_func, batched = True)

# ==========================================
# 4. EXECUTING THE FINE-TUNE
# ==========================================
print("🔥 Initiating Supervised Fine-Tuning (SFT)...")
trainer = SFTTrainer(
    # ... (keep existing model/tokenizer/dataset arguments) ...
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, 
        warmup_steps = 50,           # <-- BUMPED for a larger dataset
        # Set max_steps to -1 to use epochs for large data
        max_steps = -1,              
        num_train_epochs = 1,        # <-- Standard for large-scale SFT
        learning_rate = 1e-4,        # <-- Slightly LOWERED for stability on 30B
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(), 
        logging_steps = 10,          # <-- BUMPED to reduce log clutter
        optim = "adamw_8bit", 
        weight_decay = 0.05,         # <-- BUMPED for better generalization
        lr_scheduler_type = "cosine", # <-- SWITCHED to Cosine for smoother decay
        seed = 3407,
        output_dir = "outputs",
        save_steps = 500,            # <-- Added checkpointing for long runs
    ),
)

trainer.train()

# ==========================================
# 5. SAVING THE COMPETITION SUBMISSION
# ==========================================
print("💾 Saving LoRA Adapter for Kaggle Submission...")
model.save_pretrained("nemotron_reasoning_lora_submission")
tokenizer.save_pretrained("nemotron_reasoning_lora_submission")
print("✅ Training Complete! Your LoRA adapter is ready.")