from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login
import os

# Use environment variable for token security
token = os.getenv("HUGGINGFACE_TOKEN")
if token:
    login(token=token)

# Step 1: Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen3-1.7B", trust_remote_code=True
)

# Step 2: Load PEFT adapter
model = PeftModel.from_pretrained(
    base_model, "Rustamshry/Qwen3-1.7B-finance-reasoning"
)

# ✅ Step 3: Merge adapter into the base model
model = model.merge_and_unload()

# Step 4: Save full model
model.save_pretrained("qwen3_merged", safe_serialization=False)
tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-1.7B", trust_remote_code=True)
tokenizer.save_pretrained("qwen3_merged")
