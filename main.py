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
watermark_params, sampler_params, model_params, general_params = config_parser(config_file)

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

input_filepath = os.path.join(os.getcwd(), general_params['input_directory'], general_params['input_filename'])
with open(input_filepath, 'r') as f:
	initial_text = f.read()
token_ids = tokenizer.encode(initial_text, return_tensors='pt')[0]
input_tokens = token_ids

# generate N more tokens
for i in tqdm(range(model_params['num_output_tokens'])):

	# pass tokens through the model to get logits
	output = model(token_ids)["logits"][-1,:]

	# sample from the logits
	token_ids_np = token_ids.data.cpu().numpy()
	
    # watermark logits
	marked_output = marker.watermark(output.data.cpu().numpy(), token_ids_np)

    # sample
	tok = samp(marked_output, token_ids_np)

	# add the resulting token id to our list
	token_ids_np = np.append(token_ids_np, tok)
	token_ids = torch.from_numpy(token_ids_np)

# print out the decoded text
print(tokenizer.decode(token_ids))

# Detect Watermark
input_tokens = input_tokens.data.cpu().numpy()
output_tokens = token_ids_np[len(input_tokens):]
vocab_size = 50257 #TODO: Make this dynamic
z_metric, green_list_count = marker.detect(vocab_size, input_tokens, output_tokens)
decode_input_tokens = tokenizer.decode(input_tokens)
decode_output_tokens = tokenizer.decode(output_tokens)
print(f"Z Metric: {z_metric}")
print(f"{green_list_count} of {len(output_tokens)} tokens generated are from the green list of tokens.")
print(f"Watermark detected") if z_metric > watermark_params['z_threshold'] else print(f"Watermark NOT detected")

# Save Output
output_filename = f"output_{general_params['input_filename']}"
output_filepath = os.path.join(os.getcwd(), general_params['output_directory'], output_filename)
with open(output_filepath, 'w') as f:
	f.write(f"Input:\n{decode_input_tokens}\n\n")
	f.write(f"Output:\n{decode_output_tokens}\n\n")
	f.write(f"Z Metric: {z_metric}\n")
	f.write(f"{green_list_count} of {len(output_tokens)} tokens generated are from the green list of tokens.\n")
	if z_metric > watermark_params['z_threshold']:
		f.write(f"Watermark detected")
	else:
		f.write(f"Watermark NOT detected")