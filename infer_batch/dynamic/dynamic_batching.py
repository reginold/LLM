# "/import/mlcp-sc-nlp/llama-3_1/Meta-Llama-3.1-8B-Instruct"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading
import time
from queue import Queue

# Step 1: Load the tokenizer and LLaMA 3.8.1 model
model_name = "/import/mlcp-sc-nlp/llama-3_1/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Assign a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model.eval()

# Step 2: Define a request queue and batch processing parameters
request_queue = Queue()
MAX_BATCH_SIZE = 4  # Maximum number of requests per batch
BATCH_TIMEOUT = 0.05  # Timeout in seconds for batching

# Lock for synchronizing batch collection
queue_lock = threading.Lock()

# Step 3: Define a function for dynamic batching
def process_batch():
    while True:
        time.sleep(BATCH_TIMEOUT)  # Check for batches at regular intervals
        batch_requests = []
        with queue_lock:
            # Collect up to MAX_BATCH_SIZE requests from the queue
            while not request_queue.empty() and len(batch_requests) < MAX_BATCH_SIZE:
                batch_requests.append(request_queue.get())

        if not batch_requests:
            continue  # No requests to process

        # Extract texts from batch requests
        input_texts = [req["text"] for req in batch_requests]

        # Tokenize inputs
        tokenized_inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,  # Pad to the longest sequence in the batch
            truncation=True
        )

        # Move tensors to GPU
        input_ids = tokenized_inputs["input_ids"].to("cuda")
        attention_mask = tokenized_inputs["attention_mask"].to("cuda")

        # Perform inference
        with torch.no_grad():
            generated_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,  # Adjust max length as needed
                num_return_sequences=1,
                do_sample=False  # Greedy decoding
            )

        # Decode outputs
        decoded_outputs = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in generated_outputs
        ]

        # Record batch size and send responses back to the original requesters
        batch_size = len(batch_requests)  # Calculate batch size
        for i, req in enumerate(batch_requests):
            req["response"] = decoded_outputs[i]
            req["queue_time"] = time.time() - req["enqueue_time"]  # Calculate queue time
            req["batch_size"] = batch_size  # Record the batch size
            req["event"].set()  # Notify the requester thread that the response is ready

# Step 4: Start the batch processing thread
batching_thread = threading.Thread(target=process_batch, daemon=True)
batching_thread.start()

# Step 5: Define a function to handle individual requests
def handle_request(text):
    event = threading.Event()
    request = {
        "text": text,
        "event": event,
        "response": None,
        "enqueue_time": time.time(),  # Record enqueue time
        "queue_time": None,
        "batch_size": None,
    }
    with queue_lock:
        request_queue.put(request)
    event.wait()  # Wait for the response to be processed
    return request["response"], request["queue_time"], request["batch_size"]

# Step 6: Simulate incoming requests
if __name__ == "__main__":
    # Example user requests
    user_requests = [
        "What is AI?",
        "Explain the concept of machine learning.",
        "What is the capital of France?",
        "Tell me about space exploration.",
    ]

    # Simulate concurrent requests
    results = []
    queue_times = []
    batch_sizes = []

    def request_thread(text, results_list, queue_time_list, batch_size_list):
        response, queue_time, batch_size = handle_request(text)
        results_list.append(response)
        queue_time_list.append(queue_time)
        batch_size_list.append(batch_size)

    threads = []
    for req in user_requests:
        t = threading.Thread(
            target=request_thread,
            args=(req, results, queue_times, batch_sizes)
        )
        threads.append(t)
        t.start()
        time.sleep(0.01)  # Simulate staggered arrival of requests

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Display results, queue times, and batch sizes
    print("\nResults:")
    for i, (output, queue_time, batch_size) in enumerate(zip(results, queue_times, batch_sizes)):
        print(f"Input: {user_requests[i]}")
        print(f"Output: {output}")
        print(f"Queue Time: {queue_time:.4f} seconds")
        print(f"Batch Size: {batch_size}")
        print("-" * 50)
