from transformers import Wav2Vec2ForSequenceClassification


class AudioTransformer:
    def __init__(self):
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "facebook/wav2vec2-base-960h",
            num_labels=num_classes
        )

    def process(self, audio):
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        return self.model(**inputs)