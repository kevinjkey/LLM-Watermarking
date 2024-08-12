
import torch
import numpy as np

'''
Class implementing a sampler for inference on a model. Given the raw logits from
an LLM model, this will sample the next token id.
'''
class Sampler:

	def __init__(
		self,
		top_k=None,
		top_p=None,
		frequency_penalty=1.0,
		presence_penalty=1.0
	):
		'''
		param top_k : (None or int)
			If specified, only the top k logits should be used during sampling
			If this is specified, top_p should be None

		param top_p : (None or int)
			If specified, only the logits representing the probability mass p should be used during sampling.
			Or, if the top token has mass greater than p, the top token is returned.
			If this is specified, top_k should be None

		If top_k and top_p are both None, sample from the whole distribution (same as top_p=1.0)

		param frequency_penalty : (float)
			A penalty applied to tokens that have previously occured in the sequence. Along with
			presence_penalty, this adjusts the per-token softmax temperature.
			A penalty of 1.0 indicates no change from normal softmax.

		param presence_penalty : (float)
			A penalty applied to tokens IF they have previously occured in the sequence. Along with
			frequency_penalty, this adjusts the per-token softmax temperature.
			A penalty of 1.0 indicates no change from normal softmax.
		'''
		self.top_k = top_k
		self.top_p = top_p
		self.frequency_penalty = frequency_penalty
		self.presence_penalty = presence_penalty

		self.use_top_k = False
		self.use_top_p = False

		# Set a strategy with some boolean logic
		if self.top_k is None and self.top_p is None:
			self.top_p = 1.0
			self.use_top_p = True
		elif self.top_k is not None:
			self.use_top_k = True
			assert self.top_p is None, "Invalid strategy option. Please define only 1 of top_k or top_p or set both to None."
		elif self.top_p is not None:
			self.use_top_p = True
			assert self.top_k is None, "Invalid strategy option. Please define only 1 of top_k or top_p or set both to None."


	def sample_token(self, raw_unsorted_logits, previous_token_ids):
		'''
		param: raw_unsorted_logits (float numpy array)
			A one dimensional list of logits representing an unnormalized distribution over next tokens
			These are "unsorted" in the sense that their order aligns with vocabulary order, not with probability.

		param: previous_token_ids (int numpy array)
			A one dimensional list of ids representing the previous tokens, for calculating repetition penalties.

		returns: a single token id (integer), sampled according to the specified sampling parameters
		'''

		# Create Numpy arrays
		raw_unsorted_logits = np.array(raw_unsorted_logits)
		previous_token_ids = np.array(previous_token_ids)

		# Shift Logits
		raw_unsorted_logits = raw_unsorted_logits - np.min(raw_unsorted_logits)

		# Penalties
		k = np.ones(len(raw_unsorted_logits))

		# Count Occurrences and accrue a presence penalty
		occurrences = []
		accrued_pres_penalty = []
		for i in range(len(k)):
			num_occurences = sum(previous_token_ids==i)
			occurrences.append(num_occurences)
			if num_occurences > 0:
				accrued_pres_penalty.append(self.presence_penalty-1.0)
			else:
				accrued_pres_penalty.append(0)
		occurrences = np.array(occurrences)
		accrued_pres_penalty = np.array(accrued_pres_penalty)

		# Frequence Penalty
		accrued_freq_penalty = occurrences * (self.frequency_penalty-1.0)
		total_penalty = k + accrued_freq_penalty + accrued_pres_penalty
		
		# Adjust logits and take softmax
		adjusted_logits = raw_unsorted_logits / total_penalty
		unsorted_logits = torch.nn.functional.softmax(torch.tensor(adjusted_logits), dim=0).data.cpu().numpy()

		# Sort Logits
		sort_idx = np.argsort(unsorted_logits)
		sort_idx = sort_idx[::-1] # need to reverse indices to get descending order

		# Top K Strategy
		if self.use_top_k:
			selected_idx = sort_idx[:self.top_k]
			selected_logits = unsorted_logits[selected_idx]

			# Normalize logits to sum to 1
			selected_logits = selected_logits / np.sum(selected_logits)

			# Select token
			return np.random.choice(selected_idx, p=selected_logits)
		
		# Top P Strategy
		if self.use_top_p:
			probability_mass = 0
			idx = 0
			selected_idx = []
			selected_logits = []

			# Get tokens and logits that meet the top_p threshold
			if self.top_p==1.0:
				# Select token
				return np.random.choice(sort_idx, p=unsorted_logits[sort_idx])
			else:
				while probability_mass < self.top_p:
					probability_mass = probability_mass + unsorted_logits[sort_idx][idx]
					selected_idx.append(sort_idx[idx])
					selected_logits.append(unsorted_logits[sort_idx][idx])
					idx = idx + 1

				selected_idx = np.array(selected_idx)
				selected_logits = np.array(selected_logits)

				# Normalize logits to sum to 1
				selected_logits = selected_logits / np.sum(selected_logits)

				# Select token
				return np.random.choice(selected_idx, p=selected_logits)

	# an alternative way to call sample_token(), for convenience
	def __call__(self, raw_unsorted_logits, previous_token_ids):
		return self.sample_token(raw_unsorted_logits, previous_token_ids)




if __name__ == "__main__":
    
    # example of using this with dummy data
	
	sampler = Sampler(top_p=0.8, frequency_penalty=1.1, presence_penalty=1.1)
	# sampler = Sampler(top_k=10, frequency_penalty=1.1, presence_penalty=1.1)
	
	sequence = [1,2,3,4,5]
	
	for i in range(10):
	# for i in range(1):
		# fake logits for a vocab of size 500
		logits = np.random.randn(500)
		
		# get next token in sequence
		next_token = sampler(logits, sequence)
		sequence.append(next_token)
		
	print(sequence)