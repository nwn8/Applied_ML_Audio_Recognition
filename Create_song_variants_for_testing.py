import librosa
from pydub import AudioSegment
import numpy as np
import os

def create_song_variants_mp3(input_path, output_dir="uploaded_song"):
    # Load audio
    y, sr = librosa.load(input_path, sr=None)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create variants
    variants = {
        "slowed_10": librosa.effects.time_stretch(y, rate=0.90),
        "sped_10": librosa.effects.time_stretch(y, rate=1.10),
        "pitch_down_2": librosa.effects.pitch_shift(y, sr=sr, n_steps=-2),
        "pitch_up_2": librosa.effects.pitch_shift(y, sr=sr, n_steps=2),
    }

    # Save each variant as MP3
    for name, audio in variants.items():
        # Convert numpy array → AudioSegment
        audio_segment = AudioSegment(
            (audio * 32767).astype(np.int16).tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )

        output_path = os.path.join(output_dir, f"{name}.mp3")
        audio_segment.export(output_path, format="mp3")
        print(f"Saved: {output_path}")

    print("All MP3 variants created successfully.")



