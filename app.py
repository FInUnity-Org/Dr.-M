

# You can skip this if already logged in


from flask import Flask, request, jsonify
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
app = Flask(__name__)
login(token="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")

# ✅ Login if needed — you can skip if you've already done "huggingface-cli login" before
# login(token="hf_xxx...")

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-1.7B", trust_remote_code=True)

# ✅ Load base model with correct device map
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen3-1.7B",
    device_map="cpu",    # correct for M1, uses CPU and/or MPS
    trust_remote_code=True,
)

# ✅ Load PEFT adapter
model = PeftModel.from_pretrained(
    base_model,
    "Rustamshry/Qwen3-1.7B-finance-reasoning"
)
model.eval()

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400

        # Format prompt as chat template
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Prepare input and move to MPS (Apple GPU)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Generate
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            do_sample=True,
        )

        # Decode
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return jsonify({"response": output_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)
