import whisper

class VoiceToText:
    def __init__(self):
        self.model = whisper.load_model('base')

    def transcribe(self, file_path: str) -> str:
        result = self.model.transcribe(file_path, fp16=False, language='English')
        return result['text']

model = VoiceToText()
result = model.transcribe('sample.mp3')
print(result)