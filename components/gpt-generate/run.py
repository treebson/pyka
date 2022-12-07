from transformers import pipeline

prompt = input('\nPrompt:\n')

generator = pipeline('text-generation', model='gpt2', pad_token_id=50256)

sentences = generator(prompt, do_sample=True, top_k=50, temperature=0.6, max_length=128, num_return_sequences=3)

print('\nResponse:')
print(sentences[0]['generated_text'])

