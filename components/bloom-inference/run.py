from huggingface_hub.inference_api import InferenceApi

token = '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'
inference = InferenceApi(repo_id="bigscience/bloom", token=token)

max_new_tokens = 80
top_k = 0
num_beams = 5
no_repeat_ngram_size = 2
top_p = 0.9
seed = 42
temperature = 0.2
greedy_decoding = False

top_k = None if top_k == 0 else top_k
do_sample = False if num_beams > 0 else not greedy_decoding
num_beams = None if (greedy_decoding or num_beams == 0) else num_beams
no_repeat_ngram_size = None if num_beams is None else no_repeat_ngram_size
top_p = None if num_beams else top_p
early_stopping = None if num_beams is None else num_beams > 0

params = {
    'max_new_tokens': max_new_tokens,
    'top_k': top_k,
    'top_p': top_p,
    'temperature': temperature,
    'do_sample': do_sample,
    'seed': seed,
    'early_stopping': early_stopping,
    'no_repeat_ngram_size': no_repeat_ngram_size,
    'num_beams': num_beams
}

with open('sample.txt', 'r') as file:
    prompt = file.read()

result = inference(prompt, params=params)

print(result[0]['generated_text'])