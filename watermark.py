import numpy as np
import math

class Watermark():
    def __init__(self, hash_value, green_list_size=.5, hardness=2):
        self.hash_value=hash_value
        self.green_list_size = green_list_size
        self.hardness = hardness
    
    def _compute_green_list(self, previous_token_id, vocab_size):
        # Compute Hash Function
        random_seed = previous_token_id* self.hash_value

        # Seed RNG
        rng = np.random.default_rng(random_seed)

        # Shuffle Idxs
        random_idx = np.arange(vocab_size)
        rng.shuffle(random_idx)

        # Select green list tokens
        num_green_list_tokens = int(self.green_list_size*vocab_size)

        return random_idx[:num_green_list_tokens]
    

    def _compute_z_metric(self, green_list_count, num_output_tokens):
        
        return ((green_list_count - (self.green_list_size*num_output_tokens))/math.sqrt(num_output_tokens*self.green_list_size*(1-self.green_list_size)))


    def watermark(self, raw_logits, previous_token_ids):

        # Compute Green List of Tokens
        green_list = self._compute_green_list(previous_token_ids[-1], len(raw_logits))

        # Apply hardness to green tokens
        raw_logits[green_list] = raw_logits[green_list] + self.hardness
        
        return raw_logits

    
    def detect(self, vocab_size, input_token_ids, output_token_ids):

        # Count number of tokens appearing in the green list
        previous_token_id = input_token_ids[-1]
        green_list_count = 0

        for token in output_token_ids:

            # Compute Green List
            green_list = self._compute_green_list(previous_token_id, vocab_size)

            if token in green_list:
                green_list_count = green_list_count + 1

            previous_token_id = token

        # Compute z metric
        return self._compute_z_metric(green_list_count, len(output_token_ids)), green_list_count