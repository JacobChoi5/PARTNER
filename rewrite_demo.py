import warnings
warnings.filterwarnings("ignore", message="The attention mask is not set and cannot be inferred from input because pad token is same as eos token.*")
import torch
from transformers import AutoTokenizer
from src.lsp_model_rl.modeling_gpt2 import GPT2LMHeadModel

# Path to your trained checkpoint

CHECKPOINT_PATH = 'output/GPT2.1e-05.2.0gpu.2025-08-22012530/GP2-pretrain-step-1000.pkl'
MODEL_NAME = 'microsoft/DialoGPT-medium'
# TOKENIZER_PATH = 'dataset/sample_data.128len.db/'

# Load tokenizer and model
# print('Loading tokenizer and model...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
special_tokens = ['<SPLIT>', '<START>', '<END>']
tokenizer.add_tokens(special_tokens)
vocab_size = len(tokenizer)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(vocab_size)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
model.eval()



# Read seeker post from terminal
seeker_post = input("Enter the message to rewrite: ")
# Add EOS token to input for better separation
input_text = seeker_post + tokenizer.eos_token
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate rewritten response with more diverse settings (no attention_mask)
outputs = model.generate(
    input_ids=input_ids,
    max_length=128,
    do_sample=True,
    temperature=1.0,
    top_p=0.9,
    top_k=50,
    # pad_token_id=tokenizer.eos_token_id
)
rewritten_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# Remove the input message from the output if present
print("Input message:", seeker_post)
print("Rewritten response:", rewritten_response)
