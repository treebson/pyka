from transformers import pipeline

text = input('\nPrompt:\n')

# Allocate a pipeline for sentiment-analysis
model = pipeline('sentiment-analysis')

result = model(text)
print('')
print(result)