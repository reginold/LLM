# "/import/mlcp-sc-nlp/llama-3_1/Meta-Llama-3.1-8B-Instruct"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Step 1: Load the tokenizer and LLaMA 3.1 model
model_name = "/import/mlcp-sc-nlp/llama-3_1/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Assign a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model.eval()

# Step 2: Define input requests
requests = [
    "What is AI?",
    "Explain the concept of machine learning.",
    "What is the capital of France?",
]

# Step 3: Format inputs with a clear instruction prefix
formatted_requests = [f"Question: {req}\nAnswer:" for req in requests]

# Step 4: Tokenize inputs with padding
tokenized_inputs = tokenizer(
    formatted_requests,
    return_tensors="pt",
    padding=True,  # Automatically pad to the longest sequence in the batch
    truncation=True
)

# Move tensors to GPU
input_ids = tokenized_inputs["input_ids"].to("cuda")
attention_mask = tokenized_inputs["attention_mask"].to("cuda")

print("Input IDs with Padding:")
print(input_ids)

# Step 5: Perform Inference using `generate`
with torch.no_grad():
    generated_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=50,  # Adjust max length as needed
        num_return_sequences=1,
        do_sample=False,  # Greedy decoding
    )

# Step 6: Decode the Output
decoded_outputs = [
    tokenizer.decode(output, skip_special_tokens=True) for output in generated_outputs
]

# Step 7: Display Results
print("\nResults:")
for i, output in enumerate(decoded_outputs):
    print(f"Input: {requests[i]}")
    print(f"Output: {output}")
    print("-" * 50)