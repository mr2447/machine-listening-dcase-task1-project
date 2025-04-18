import os
import glob
import numpy as np
from dcase_util.features import FeatureExtractor
from dcase_util.containers import AudioContainer

AUDIO_DIR = 'data/TAU-urban-acoustic-scenes-2022-mobile-development/audio'
OUTPUT_DIR = 'features/audio.2'

os.makedirs(OUTPUT_DIR, exist_ok=True)

fe = FeatureExtractor(
    fs=44100,
    win_length_seconds=0.04,
    hop_length_seconds=0.02,
    n_mels=40,
    fmin=20,
    fmax=22050,
    spectrogram_type='magnitude',
    normalize_mel_bands=True,
    htk_compat=True,
)

wav_files = glob.glob(os.path.join(AUDIO_DIR, '*.wav'))
print(f"🎧 Found {len(wav_files)} .wav files in {AUDIO_DIR}")

for file_path in wav_files:
    try:
        print(f"🔍 Processing: {file_path}")
        audio = AudioContainer().load(file_path, fs=44100)
        print(f"📏 Audio loaded: {audio}, shape: {None if audio.data is None else audio.data.shape}")
        features = fe.extract(audio)  # Should return shape (40, N)

        if features.shape[0] != 40 or features.shape[1] < 51:
            print(f"⚠️ Skipping {file_path} — bad shape: {features.shape}")
            continue

        # Trim or pad to (40, 51)
        features = features[:, :51]

        filename = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f'{filename}.npy')
        np.save(output_path, features)
        print(f"✅ Saved features: {output_path}")

    except Exception as e:
        print(f"❌ Failed to process {file_path}: {e}")