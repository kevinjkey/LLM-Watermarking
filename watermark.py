import numpy as np

class Watermark():
    def __init__(self, hash_value, green_list_size=.5, hardness=2):
        self.hash_value=hash_value
        self.green_list_size = green_list_size
        self.hardness = hardness

    def watermark(self, raw_logits, previous_token_ids):

        # Compute Hash Function
        random_seed = previous_token_ids[-1] * self.hash_value

        # Seed RNG
        rng = np.random.default_rng(random_seed)

        # Shuffle Idxs
        random_idx = np.arange(len(raw_logits))
        # print(random_idx)
        rng.shuffle(random_idx)
        # print(random_idx)

        # Select green list tokens to apply hardness
        num_green_list_tokens = int(self.green_list_size*len(raw_logits))
        # print("Num Green Tokens: ", num_green_list_tokens)
        random_idx = random_idx[:num_green_list_tokens]
        # print(raw_logits[random_idx])
        raw_logits[random_idx] = raw_logits[random_idx] + self.hardness
        # print(raw_logits[random_idx])

        # print(kevin)

        #arange len logits, shuffle, top %, add hardness to logits[selected], sample
        
        return raw_logits
        # return np.random.randint(6, 30)

    
    