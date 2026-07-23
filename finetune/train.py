# =============================================================================
# finetune/train.py — QLoRA fine-tuning for qwen3:4b on Google Colab
# Run this on Google Colab with GPU (T4 or better)
# Upload dataset.jsonl to Colab before running
# =============================================================================

# ── Step 1: Install dependencies (run in Colab cell) ──────────────────────────
# !pip install unsloth
# !pip install torch transformers datasets peft trl

# ── Step 2: Import ─────────────────────────────────────────────────────────────
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
import json

# ── Step 3: Configuration ──────────────────────────────────────────────────────
MODEL_NAME   = "unsloth/Qwen3-4B-unsloth-bnb-4bit"  # base model
OUTPUT_DIR   = "./iot_ids_qwen3_finetuned"
DATASET_PATH = "./dataset.jsonl"                      # upload this to Colab
MAX_SEQ_LEN  = 1024
EPOCHS       = 3
BATCH_SIZE   = 2
LR           = 2e-4

# ── Step 4: Load base model with 4-bit quantization ───────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LEN,
    dtype          = None,        # auto-detect
    load_in_4bit   = True,        # QLoRA
)

# ── Step 5: Apply LoRA adapters ───────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,           # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    lora_alpha     = 16,
    lora_dropout   = 0.05,
    bias           = "none",
    use_gradient_checkpointing = True,
)

# ── Step 6: Load and format dataset ───────────────────────────────────────────
def format_prompt(example):
    """Format as Alpaca-style instruction-response."""
    text = f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
    return {"text": text}

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(format_prompt)

print(f"Dataset size: {len(dataset)} samples")
print(f"Sample:\n{dataset[0]['text'][:300]}")

# ── Step 7: Training arguments ────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir              = OUTPUT_DIR,
    num_train_epochs        = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = 4,
    learning_rate           = LR,
    fp16                    = not torch.cuda.is_bf16_supported(),
    bf16                    = torch.cuda.is_bf16_supported(),
    logging_steps           = 10,
    save_steps              = 100,
    warmup_ratio            = 0.1,
    lr_scheduler_type       = "cosine",
    report_to               = "none",
)

# ── Step 8: Train ─────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = dataset,
    dataset_text_field = "text",
    max_seq_length  = MAX_SEQ_LEN,
    args            = training_args,
)

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete.")

# ── Step 9: Save fine-tuned model ─────────────────────────────────────────────
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved → {OUTPUT_DIR}")

# ── Step 10: Export to GGUF for Ollama ────────────────────────────────────────
# Run this after training to convert for local use:
# model.save_pretrained_gguf(OUTPUT_DIR + "_gguf",
#                            tokenizer,
#                            quantization_method="q4_k_m")
# Then in terminal:
# ollama create iot-ids-llm -f Modelfile
# Where Modelfile contains: FROM ./iot_ids_qwen3_finetuned_gguf/model.gguf

print("""
==============================================
Next steps after training:
1. Download the OUTPUT_DIR folder from Colab
2. Convert to GGUF: uncomment save_pretrained_gguf above
3. Create Ollama model:
   echo 'FROM ./model.gguf' > Modelfile
   ollama create iot-ids-llm -f Modelfile
4. Update config.py:
   LLM_MODEL = 'iot-ids-llm'
==============================================
""")
