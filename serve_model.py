from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./src/fine-tuned-gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('./src/fine-tuned-gpt2')

# Set the pad_token to eos_token or add a custom pad token
tokenizer.pad_token = tokenizer.eos_token


# Alternatively, you can add a custom padding token
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# model.resize_token_embeddings(len(tokenizer))

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json

    prompt = data.get("prompt", "")
    max_length = data.get("max_length", 1000)

    inputs = tokenizer.encode_plus(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id
    )
    print(data)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(tokenizer)
    return jsonify({"generated_text": generated_text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
