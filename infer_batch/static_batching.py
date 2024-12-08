import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Step 1: Load a tokenizer and a model
model_path = "/import/mlcp-sc-nlp/llama-3_1/Meta-Llama-3.1-8B-Instruct"  # Example model
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Check if the tokenizer has a padding token, if not, add one
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

model = AutoModelForCausalLM.from_pretrained(model_path)
model.resize_token_embeddings(len(tokenizer))  # Resize the model embeddings to include the new padding token
model.eval()

# Step 2: Define input requests
requests = [
    "What is AI?",
    "Explain the concept of machine learning.",
    "What is the capital of France?",
]

# Step 3: Tokenize inputs
# Convert text into token IDs, and calculate the maximum sequence length
tokenized_inputs = [tokenizer(req, return_tensors="pt", padding=True, truncation=True) for req in requests]

# Step 4: Static Batching
# Pad all inputs to the maximum sequence length in the batch
max_length = max(input_data['input_ids'].shape[1] for input_data in tokenized_inputs)
batch_input_ids = torch.stack([torch.nn.functional.pad(input_data['input_ids'], (0, max_length - input_data['input_ids'].shape[1])) for input_data in tokenized_inputs])
batch_attention_mask = torch.stack([torch.nn.functional.pad(input_data['attention_mask'], (0, max_length - input_data['attention_mask'].shape[1])) for input_data in tokenized_inputs])

print("Batch Input IDs:")
print(batch_input_ids)

print("Batch Attention Mask:")
print(batch_attention_mask)

# Print the shapes of the batch input IDs and attention mask
print("Batch Input IDs Shape:", batch_input_ids.shape)
print("Batch Attention Mask Shape:", batch_attention_mask.shape)

# Ensure the shapes are compatible with the model's expected input
# For example, if the model expects input of shape [batch_size, sequence_length]
batch_input_ids = batch_input_ids.squeeze(1)
batch_attention_mask = batch_attention_mask.squeeze(1)

# Step 5: Perform Inference
# Forward pass through the model
with torch.no_grad():
    outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)

# Step 6: Decode the Output
# Get the logits (predictions) and decode the text for each input in the batch
logits = outputs.logits
predictions = torch.argmax(logits, dim=-1)

decoded_outputs = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]

# Step 7: Display Results
print("\nResults:")
for i, output in enumerate(decoded_outputs):
    print(f"Input: {requests[i]}")
    print(f"Output: {output}")
    print("-" * 50)