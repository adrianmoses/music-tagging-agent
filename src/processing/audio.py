def process_audio(file_path):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=44100)

    # Normalize audio
    audio = librosa.util.normalize(audio)

    return audio, sr


class AudioModelPipeline:
    def __init__(self):
        self.cnn = load_cnn_model()
        self.transformer = load_transformer_model()

    def process(self, features):
        cnn_output = self.cnn(features['mfcc'])
        temporal_features = self.prepare_temporal_features(features)
        transformer_output = self.transformer(temporal_features)

        return {
            'genre': cnn_output['genre_predictions'],
            'mood': transformer_output['mood_predictions'],
            'instruments': cnn_output['instrument_predictions']
        }