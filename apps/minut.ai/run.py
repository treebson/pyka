import whisper
from huggingface_hub.inference_api import InferenceApi

# Config
target_path = 'data/sample_4.mp3'
token = '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'
params = {
    'max_new_tokens': 150,
    'top_k': None,
    'top_p': None,
    'temperature': 0.22,
    'do_sample': False,
    'seed': 420,
    'early_stopping': None,
    'no_repeat_ngram_size': 2,
    'num_beams': 5
}

# From Open AI's 'whisper' package
class Transcriber:
    def __init__(self):
        self.model = whisper.load_model('base')

    def transcribe(self, file_path: str) -> str:
        result = self.model.transcribe(file_path, fp16=False, language='English')
        return result['text']

# Instantiate models
print('\nInstantiating models...')
print('- whisper')
transcriber = Transcriber()
print('- bloom')
bloom = InferenceApi(repo_id="bigscience/bloom", token=token)

# Transcribe with whisper
print("\nTranscribing with _whisper_")
text = transcriber.transcribe(target_path)

# Prompt engineering
prompt = f'''
**Meeting Transcription:**
{text}

**Questions:**
1. What is the subject of this meeting (1 sentence)?
2. What were the themes discussed?
3. What are the action items (if any)?

**Answers:**
'''
print('Prompting bloom...')
result = bloom(prompt, params=params)
result = result[0]['generated_text']

print(result)