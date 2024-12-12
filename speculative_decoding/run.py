import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
DRAFT_MODEL_NAME = "/import/ml-sc-nlpcheckpoints-scratch3/bol/llama_3d2/meta-llama-Llama-3.2-3B-Instruct/"
# TARGET_MODEL_NAME = "/import/ml-sc-nlpcheckpoints-scratch3/bol/Meta-Llama-3.1-70B-Instruct"
TARGET_MODEL_NAME = "/import/ml-sc-nlpcheckpoints-scratch3/bol/llama_3d2/meta-llama-Llama-3.2-3B-Instruct/"

# Load Models and Tokenizers
draft_tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL_NAME)
draft_model = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL_NAME).cuda()

target_tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_NAME).cuda()

def speculative_decoding(prompt, max_length=50, acceptance_rate=0.9):
    """
    Implements speculative decoding where the draft model generates tokens that
    are verified and refined by the target model. Tokens are accepted based
    on a predefined acceptance rate.

    Args:
        prompt (str): Input text prompt.
        max_length (int): Maximum length of generated tokens.
        acceptance_rate (float): Probability threshold for accepting draft tokens.

    Returns:
        str: Final decoded sequence.
    """
    # Encode input prompt
    input_ids = draft_tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    # Generate initial tokens with draft model
    with torch.no_grad():
        draft_outputs = draft_model.generate(input_ids, max_length=max_length, do_sample=True)

    draft_sequence = draft_tokenizer.decode(draft_outputs[0], skip_special_tokens=True)

    # Feed draft sequence to target model for refinement
    target_input_ids = target_tokenizer(draft_sequence, return_tensors="pt").input_ids.cuda()

    with torch.no_grad():
        target_outputs = target_model.generate(
            target_input_ids, max_length=max_length, do_sample=True, eos_token_id=target_tokenizer.eos_token_id
        )

    # Decode and compare outputs
    draft_tokens = draft_tokenizer.convert_ids_to_tokens(draft_outputs[0])
    target_tokens = target_tokenizer.convert_ids_to_tokens(target_outputs[0])

    refined_tokens = []
    for draft_token, target_token in zip(draft_tokens, target_tokens):
        if torch.rand(1).item() < acceptance_rate:
            refined_tokens.append(draft_token)
        else:
            refined_tokens.append(target_token)

    refined_sequence = target_tokenizer.convert_tokens_to_string(refined_tokens)

    return refined_sequence

if __name__ == "__main__":
    # Example usage
    prompt = "The future of AI is"
    result = speculative_decoding(prompt, max_length=512, acceptance_rate=0.9)
    print("Generated Text:", result)
