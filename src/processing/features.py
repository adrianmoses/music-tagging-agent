class AudioFeatureExtractor:
    def extract_features(self, audio, sr):
        features = {
            'mfcc': librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13),
            'spectral_centroid': librosa.feature.spectral_centroid(y=audio, sr=sr),
            'chroma': librosa.feature.chroma_stft(y=audio, sr=sr),
            'tempo': librosa.beat.tempo(y=audio, sr=sr),
            'onset_env': librosa.onset.onset_strength(y=audio, sr=sr),
            'harmonic': librosa.effects.harmonic(y=audio)
        }
        return features


