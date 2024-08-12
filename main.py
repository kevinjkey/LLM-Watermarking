import numpy as np
import torch
from tqdm import tqdm

'''
This script uses the Sampler class to sample text from GPT2-small.
'''

from transformers import AutoTokenizer, AutoModelForCausalLM
from sampler import Sampler
from watermark import Watermark

samp = Sampler(top_k=30, frequency_penalty=1.1, presence_penalty=1.2)
marker = Watermark(43, .5, 2)

# download gpt2 and the associated tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large")
model.eval()

#TODO: READ IN TEXT
intial_text = "Thomas Jefferson was the"
token_ids = tokenizer.encode(intial_text, return_tensors='pt')[0]
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