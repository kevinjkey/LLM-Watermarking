import numpy as np
import torch
from tqdm import tqdm
import os

'''
This script uses the Sampler class to sample text from GPT2-large.
'''

from transformers import AutoTokenizer, AutoModelForCausalLM
from sampler import Sampler
from watermark import Watermark
from utils.import_config import config_parser

# Parse Configuration File
config_file = os.path.join(os.getcwd(), 'config.yaml')
watermark_params, sampler_params, model_params = config_parser(config_file)

# Create the Watermarker
marker = Watermark(hash_value=watermark_params['hash_value'],
				   green_list_size = watermark_params['green_list_size'],
				   hardness = watermark_params['hardness'])

# Create the Sampler
samp = Sampler(top_k=sampler_params['top_k'],
			   frequency_penalty=sampler_params['frequency_penalty'],
			   presence_penalty=sampler_params['presence_penalty'])

# download model and the associated tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_params['model_name'])
model = AutoModelForCausalLM.from_pretrained(model_params['model_name'])
model.eval()

#TODO: READ IN TEXT
# intial_text = "Thomas Jefferson was the"
with open('/Users/kevinkey/Documents/JHU/ChatGPT/Final/LLM-Watermarking/input/input1.txt', 'r') as f:
	initial_text = f.read()
print(initial_text)
token_ids = tokenizer.encode(initial_text, return_tensors='pt')[0]
input_tokens = token_ids
print(token_ids)

# generate N more tokens. We are not using kv cache or anything smart.
# This may be pretty slow.
for i in tqdm(range(50)):

	# pass tokens through the model to get logits
	output = model(token_ids)["logits"][-1,:]

	# sample from the logits
	token_ids_np = token_ids.data.cpu().numpy()
	
    # watermark logits
	marked_output = marker.watermark(output.data.cpu().numpy(), token_ids_np)

    # sample
	# tok = samp(output.data.cpu().numpy(), token_ids_np)
	tok = samp(marked_output, token_ids_np)

	# add the resulting token id to our list
	token_ids_np = np.append(token_ids_np, tok)
	token_ids = torch.from_numpy(token_ids_np)


# print out resulting ids
print(token_ids)

# print out the decoded text
print(tokenizer.decode(token_ids))

# Test detector
input_tokens = input_tokens.data.cpu().numpy()
output_tokens = token_ids[len(input_tokens):].data.cpu().numpy()
print("Input Tokens: ", input_tokens)
print("Output Tokens: ", output_tokens)
vocab_size = 50257
z_metric = marker.detect(vocab_size, input_tokens, output_tokens)