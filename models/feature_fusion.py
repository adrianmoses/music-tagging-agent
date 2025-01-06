

def prepare_temporal_features(features):
    temporal_features = {
        'onset_envelope': librosa.onset.onset_strength(
            y=features['harmonic'],
            sr=sr,
            hop_length=512
        ),
        'tempogram': librosa.feature.tempogram(
            onset_envelope=onset_envelope,
            sr=sr,
            hop_length=512
        )
    }
    return temporal_features