import numpy as np
import torch
from tqdm import tqdm
import os
import sys
import json

from transformers import AutoTokenizer
from sampler import Sampler
from watermark import Watermark

from utils.import_config import config_parser


def main():

    # Parse Configuration File
    config_file = os.path.join(os.getcwd(), 'config.yaml')
    watermark_params, sampler_params, model_params, general_params = config_parser(config_file)

    # Create the Watermarker
    marker = Watermark(hash_value=watermark_params['hash_value'],
                    green_list_size = watermark_params['green_list_size'])
    
    # Download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_params['model_name'])
    vocab_size = len(tokenizer.get_vocab())
    
    # Open the input file and tokenize
    input_filepath = os.path.join(os.getcwd(), general_params['input_directory'], general_params['input_filename'])
    with open(input_filepath, 'r') as f:
        initial_input_text = f.read()
    input_token_ids = tokenizer.encode(initial_input_text, return_tensors='pt')[0]

    # Open the output file and tokenize
    output_filepath = os.path.join(os.getcwd(), general_params['output_directory'], general_params['output_filename'])
    with open(output_filepath, 'r') as f:
        initial_output_text = f.read()
    output_token_ids = tokenizer.encode(initial_output_text, return_tensors='pt')[0]    

    # Convert tokens to Numpy arrays
    input_token_ids = np.array(input_token_ids)
    output_token_ids = np.array(output_token_ids)

    # Compute Z Metric to detect AI
    z_metric, green_list_count = marker.detect(vocab_size, input_token_ids, output_token_ids)

    decode_input_tokens = tokenizer.decode(input_token_ids)
    decode_output_tokens = tokenizer.decode(output_token_ids)
    
    # Present results
    print(f"Input: {decode_input_tokens}")
    print(f"Output: {decode_output_tokens}")
    print(f"Z Metric: {z_metric}")

    # Save Z Metric in Output file
    with open(output_filepath, 'a') as f:
        f.write(f"\n\nZ Metric: {z_metric}")
    
if __name__ == "__main__":
    main()