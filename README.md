# LLM-Watermarking
This repository implements the "soft" watermarking method from [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226). This implementation uses GPT2-XL from OpenAI Community.

## Installation
Create a `conda` environment with Python 3.10+. Using `pip`, install the `requirements.txt` file.
```
conda create -n watermarker python=3.10
conda activate watermarker
pip install -r requirements.txt
```

## Usage
### `main.py`
This script requires:
1) a `config.yaml` configuration file
1) an input text file with the prompt to be provided to the model

This script will take an input prompt and generate an output response based on the values from `config.yaml`. The output response will automatically be saved to a text file with `output` prepended to the name of the input text file. Additional information will be saved in the output file, such as the input prompt, the z metric, and whether or not the watermark was detected based on the z threshold.

`main.py` will the find `config.yaml` file if the script is run from the root directory. A sample configuration file:
```
watermarker:
  hash_value: 737800 # in production, this value would not be shared with the public as this exposes how to re-create the green list of tokens
  green_list_size: 0.25 # value between 0..1 to split the list of tokens at the output of the model before the softmax is applied. Random split using the hash value multiplied by the previous token id to seed the random split function.
  hardness: 2 # value added to raw logits in the green token list before softmax is applied
  z_threshold: 4 # threshold for determining is watermark is present or not

sampler:
  type: "top_p" # sampler type: top_p or top_k
  top_k: 15
  top_p: .9
  frequency_penalty: 1.0
  presence_penalty: 1.0

model:
  model_name: "openai-community/gpt2-xl"
  num_output_tokens: 150

general:
  input_directory: "input" # input prompt file directory. Used for main.py and ai_detector.py
  input_filename: "input.txt" 
  output_directory: "output" # output file directory. Used for main.py and ai_detector.py
  output_filename: "output.txt" # only used with ai_detector.py
```

### `ai_detector.py`
After initial input-output pairs have been generated, humans or other AI-based models can re-generate the output response. Those responses can be easily evaluated using this script and setting the appropriate input/output filenames and paths in the `config.yaml` file. The z metric will be appended to the output text file. In order for this detector to work, the hash value and green list size used at training must be used here, and the size of the original vocabular must be known.

### `watermark.py`
This class creates the software watermark and has two methods:
1) `watermark`: used during the initial output generation, raw logits are used as input to this function and the hardness value applied to the computed list of green tokens.
1) `detect`: used after an output response has been generated or re-generated by humans or AI to determine if output tokens come from the green list of tokens.

### `sampler.py`
My sampler implementation from an earlier module, with the ability to use a `top_p` or `top_k` sampling strategy and include `presence` or `frequency` penalties if desired.

## Repository Structure
### `ai_detector_test` directory
Contains the output from humans, ChatGPT, Gemini, Claude, and Llama3 attempting to re-write the output prompt generated by the original `main.py` script.

### `input` directory
Contains the original five input prompts which are:
```
PROMPT #1
Each year, Apple releases a new iPhone with many hardware upgrades. This year, 

PROMPT #2
Yesterday, I returned from my first ever trip to the island of Maui, Hawaii. The island was 

PROMPT #3
The most delicious summer meal is made with 

PROMPT #4
Family Secret Sauce Recipe

Ingredients:
1 Tomato, diced
1 Onion, chopped
1 can of water
Garlic
Extra Virgin Olive Oil
Spices, to tase

Directions:
Preheat

PROMPT #5
Education:
Bachelor of Science, Villanova University
- Honors

Skills:
Machine Learning, Python, LLMs, Docker

Experience:
Engineer, Johns Hopkins APL
- Designed 
```

### `output` directory
Contains the output generated by GPT2-XL and include the baseline z-metric.

### `utils` directory
Contains a utility function for opening and parsing the yaml configuration file.

### `IO_for_human_review.docx`
A clean way to present the input prompts and responses to humans for output response re-generation.

### `results.csv`
A table highlighting the re-generated output source, prompt number, and z score.