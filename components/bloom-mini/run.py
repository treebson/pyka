import transformers
from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast
import torch

bloom = 'bigscience/bloom-560m'
model = BloomForCausalLM.from_pretrained(bloom)
tokenizer = BloomTokenizerFast.from_pretrained(bloom)
# sizes can be found here: https://huggingface.co/docs/transformers/model_doc/bloom

with open('sample.txt', 'r') as file:
    prompt = file.read()
print('\n---- PROMPT ----\n')
print(prompt)

print('\n---- RESPONSE ----\n')
inputs = tokenizer(prompt, return_tensors='pt')
# Beam Search
print(tokenizer.decode(model.generate(inputs["input_ids"],
                       max_new_tokens=50, 
                       num_beams=5, 
                       no_repeat_ngram_size=2,
                       early_stopping=True
                      )[0]))