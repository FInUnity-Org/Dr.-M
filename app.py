from flask import Flask, request, jsonify
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = Flask(__name__)

login(token="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")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-1.7B")
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen3-1.7B", device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "Rustamshry/Qwen3-1.7B-finance-reasoning")
model.eval()


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        prompt = data.get("prompt")

        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        input_ids = tokenizer(text, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **input_ids,
            max_new_tokens=2048,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            do_sample=True,
        )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return jsonify({"response": output_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)
