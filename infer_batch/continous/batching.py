import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading
import time
from queue import Queue

# Step 1: Load the tokenizer and LLaMA 3.8.1 model
model_name = "/import/mlcp-sc-nlp/llama-3_1/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Assign a padding token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model.eval()

# Step 2: Define global KV cache and request queue
kv_cache = {}  # Simulated KV cache for the example
request_queue = Queue()

# Step 3: Continuous batching parameters
MAX_BATCH_SIZE = 4  # Maximum batch size
BATCH_TIMEOUT = 0.05  # Timeout in seconds for triggering a batch
global_lock = threading.Lock()  # Lock for KV cache and queue management

# Step 4: Continuous batching logic
def process_batch():
    while True:
        time.sleep(BATCH_TIMEOUT)  # Check for batches at regular intervals
        batch_requests = []

        with global_lock:
            # Collect requests up to MAX_BATCH_SIZE
            while not request_queue.empty() and len(batch_requests) < MAX_BATCH_SIZE:
                batch_requests.append(request_queue.get())

        if not batch_requests:
            continue

        # Step 1: Prepare inputs with contextual prefixes
        input_texts = [f"Input: {req['text']}\nOutput:" for req in batch_requests]
        tokenized_inputs = tokenizer(
            input_texts, return_tensors="pt", padding=True, truncation=True
        )

        input_ids = tokenized_inputs["input_ids"].to("cuda")
        attention_mask = tokenized_inputs["attention_mask"].to("cuda")

        # Step 2: Generate responses
        with torch.no_grad():
            generated_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=100,  # Increase max length for better response coverage
                num_return_sequences=1,
                do_sample=False  # Disable sampling for deterministic results
            )

        # Step 3: Decode outputs
        decoded_outputs = [
            tokenizer.decode(output, skip_special_tokens=True).strip()
            for output in generated_outputs
        ]

        # Step 4: Map responses back to the original requests
        for i, req in enumerate(batch_requests):
            # Extract only the portion after "Output:"
            response = decoded_outputs[i].split("Output:", 1)[-1].strip()
            req["response"] = response
            req["event"].set()  # Notify requester thread that response is ready

# Step 5: Start the continuous batching thread
batching_thread = threading.Thread(target=process_batch, daemon=True)
batching_thread.start()

# Step 6: Function to handle individual requests
def handle_request(text):
    event = threading.Event()
    request_id = str(time.time())  # Unique ID for the request
    request = {"id": request_id, "text": text, "event": event, "response": None}
    with global_lock:
        request_queue.put(request)
    event.wait()  # Wait for the batch processing
    return request["response"]

# Step 7: Simulate incoming requests
if __name__ == "__main__":
    # Example requests
    user_requests = [
        "What is AI?",
        "Explain the concept of machine learning.",
        "What is the capital of France?",
        "Tell me about space exploration.",
    ]

    results = []
    threads = []
    for req in user_requests:
        t = threading.Thread(target=lambda r, res: res.append(handle_request(r)), args=(req, results))
        threads.append(t)
        t.start()
        time.sleep(0.01)  # Simulate staggered arrival of requests

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Display results
    print("\nResults:")
    for i, output in enumerate(results):
        print(f"Input: {user_requests[i]}")
        print(f"Output: {output}")
        print("-" * 50)