from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./fine-tuned-gpt2')  # Replace with your model's directory
tokenizer = GPT2Tokenizer.from_pretrained('./fine-tuned-gpt2')

# Set the pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token


def generate_text(prompt, max_length=100, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.95, num_beams=5):
    # Tokenize the input prompt with padding and attention mask
    inputs = tokenizer.encode_plus(prompt, return_tensors="pt", padding=True, truncation=True)

    # Generate text with attention mask and beam search
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the generated text and return it
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# Example prompts
prompts = [
    "What is METEV?"
]

# Generate and print outputs for each prompt
for prompt in prompts:
    print(f"Prompt: {prompt}")
    print(f"Generated text: {generate_text(prompt)}\n")
