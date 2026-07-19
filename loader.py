"""
loader.py

Loads MP3 songs from the songs folder into the SQLite database.

For each song, this script:
1. Loads or generates lyrics using Whisper
2. Cleans lyrics and creates word statistics
3. Generates an audio fingerprint using AcoustID / Chromaprint
4. Extracts tempo and MFCC audio features using Librosa
5. Stores everything in SQLite

Expected database tables:
- songs
- fingerprints
- audio_features
"""

import os
import re
import json
import sqlite3
from collections import Counter

import acoustid
import whisper
import librosa
import numpy as np

import create_database

# -------------------------------------------------
# CREATE SONOGS_DATABASE.DB to store song data
# -------------------------------------------------

create_database.create_database()
# -------------------------------------------------
# CONFIG
# -------------------------------------------------

DB_FILE = "songs_database.db"
SONG_FOLDER = "songs"
LYRICS_FOLDER = "lyrics"

WHISPER_MODEL_SIZE = "small"
N_MFCC = 20


# -------------------------------------------------
# FOLDER SETUP
# -------------------------------------------------

os.makedirs(SONG_FOLDER, exist_ok=True)
os.makedirs(LYRICS_FOLDER, exist_ok=True)


# -------------------------------------------------
# STOPWORDS
# You can expand this later.
# -------------------------------------------------

STOPWORDS = {
    "the", "a", "an",
    "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "being",
    "i", "me", "my", "mine",
    "you", "your", "yours",
    "he", "him", "his",
    "she", "her", "hers",
    "it", "its",
    "we", "us", "our", "ours",
    "they", "them", "their", "theirs",
    "to", "of", "in", "on", "for", "with", "at", "by", "from",
    "up", "down", "out", "over", "under",
    "again", "then", "once",
    "this", "that", "these", "those",
    "am", "do", "does", "did",
    "so", "if", "as", "because",
    "just", "not", "no", "yes",
    "oh", "ah", "yeah", "yea", "hey", "hmm", "mmm"
}


# -------------------------------------------------
# DATABASE CONNECTION
# -------------------------------------------------

def get_connection():
    """
    Opens a SQLite connection and enables foreign key support.
    """

    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# -------------------------------------------------
# DUPLICATE CHECK
# -------------------------------------------------

def song_exists(cursor, file_name):
    """
    Checks whether this file has already been loaded.
    """

    cursor.execute("""
        SELECT song_id
        FROM songs
        WHERE file_name = ?
        LIMIT 1
    """, (file_name,))

    return cursor.fetchone() is not None


# -------------------------------------------------
# LYRICS: LOAD EXISTING OR GENERATE WITH WHISPER
# -------------------------------------------------

def load_or_generate_lyrics(song_name, audio_path, whisper_model):
    """
    Loads lyrics from lyrics folder if available.
    Otherwise transcribes the audio with Whisper.
    """

    lyrics_path = os.path.join(LYRICS_FOLDER, f"{song_name}.txt")

    if os.path.exists(lyrics_path):
        print(f"Using existing lyrics: {lyrics_path}")

        with open(lyrics_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    print(f"Generating lyrics with Whisper for: {song_name}")

    try:
        result = whisper_model.transcribe(audio_path)
        text = result.get("text", "").strip()

        with open(lyrics_path, "w", encoding="utf-8") as f:
            f.write(text)

        return text

    except Exception as e:
        print(f"Whisper failed for {song_name}: {e}")
        return ""


# -------------------------------------------------
# LYRICS PROCESSING
# -------------------------------------------------

def process_lyrics(text):
    """
    Cleans lyrics and returns:

    cleaned_lyrics
    word_count
    unique_word_count
    word_freq
    """

    if not text:
        return "", 0, 0, {}

    text = text.lower()

    words = re.findall(r"\b\w+\b", text)

    cleaned_words = [
        word
        for word in words
        if word not in STOPWORDS and len(word) > 1
    ]

    cleaned_lyrics = " ".join(cleaned_words)
    word_count = len(cleaned_words)
    unique_word_count = len(set(cleaned_words))
    word_freq = Counter(cleaned_words)

    return cleaned_lyrics, word_count, unique_word_count, dict(word_freq)


# -------------------------------------------------
# AUDIO FINGERPRINTING
# -------------------------------------------------

def normalize_fingerprint(fingerprint):
    """
    Makes sure the fingerprint can be safely stored as text.
    """

    if fingerprint is None:
        return None

    if isinstance(fingerprint, bytes):
        return fingerprint.decode("utf-8", errors="ignore")

    return str(fingerprint)


def fingerprint_file(file_path):
    """
    Generates an AcoustID / Chromaprint fingerprint.
    Returns duration and fingerprint.
    """

    try:
        duration, fingerprint = acoustid.fingerprint_file(file_path)
        fingerprint = normalize_fingerprint(fingerprint)

        return duration, fingerprint

    except Exception as e:
        print(f"Error fingerprinting {file_path}: {e}")
        return None, None


# -------------------------------------------------
# AUDIO FEATURE EXTRACTION
# -------------------------------------------------

def extract_audio_features(file_path):
    """
    Extracts tempo and 20 MFCC mean values from the audio file.

    Returns:
    tempo, mfcc_values

    mfcc_values is a list of 20 numbers.
    """

    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # librosa may return tempo as a numpy array depending on version
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo.item())
        else:
            tempo = float(tempo)

        mfccs = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=N_MFCC
        )

        mfcc_means = np.mean(mfccs, axis=1)

        return tempo, mfcc_means.tolist()

    except Exception as e:
        print(f"Audio feature extraction failed for {file_path}: {e}")
        return None, None


# -------------------------------------------------
# DATABASE INSERTS
# -------------------------------------------------

def insert_song(
    cursor,
    title,
    duration,
    lyrics,
    word_count,
    unique_word_count,
    word_freq,
    cleaned_lyrics
):
    """
    Inserts a song row and returns the new song_id.
    """

    cursor.execute("""
        INSERT INTO songs
        (
            title,
            duration,
            lyrics,
            word_count,
            unique_word_count,
            word_freq, 
            cleaned_lyrics
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        title,
        duration,
        lyrics,
        word_count,
        unique_word_count,
        json.dumps(word_freq),
        cleaned_lyrics
    ))

    return cursor.lastrowid


def insert_fingerprint(cursor, song_id, fingerprint):
    """
    Inserts the audio fingerprint for a song.
    """

    cursor.execute("""
        INSERT INTO fingerprints
        (
            song_id,
            fingerprint
        )
        VALUES (?, ?)
    """, (
        song_id,
        fingerprint
    ))

    return cursor.lastrowid


def insert_audio_features(cursor, song_id, tempo, mfcc_values):
    """
    Inserts tempo and 20 MFCC values into audio_features.
    """

    if mfcc_values is None or len(mfcc_values) != N_MFCC:
        raise ValueError(f"Expected {N_MFCC} MFCC values, got {len(mfcc_values) if mfcc_values else 0}")

    mfcc_columns = ", ".join([f"mfcc_{i}" for i in range(1, N_MFCC + 1)])
    mfcc_placeholders = ", ".join(["?"] * N_MFCC)

    query = f"""
        INSERT INTO audio_features
        (
            song_id,
            tempo,
            {mfcc_columns}
        )
        VALUES
        (
            ?,
            ?,
            {mfcc_placeholders}
        )
    """

    values = [song_id, tempo] + mfcc_values

    cursor.execute(query, values)

    return cursor.lastrowid


# -------------------------------------------------
# MAIN LOADER
# -------------------------------------------------

def load_songs():
    """
    Main process for loading all MP3 files from SONG_FOLDER.
    """

    files = [
        file
        for file in os.listdir(SONG_FOLDER)
        if file.lower().endswith(".mp3")
    ]
    
    print(files)

    if not files:
        print(f"No .mp3 files found in folder: {SONG_FOLDER}")
        return

    print(f"Found {len(files)} mp3 file(s).")

    print(f"Loading Whisper model: {WHISPER_MODEL_SIZE}")
    whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)

    conn = get_connection()
    cursor = conn.cursor()

    try:
        for file_name in files:
            song_name = os.path.splitext(file_name)[0]
            input_path = os.path.join(SONG_FOLDER, file_name)

            print("\n----------------------------------------")
            print(f"Processing: {song_name}")
            print("----------------------------------------")


            try:
                # -----------------------------
                # Lyrics
                # -----------------------------
                lyrics = load_or_generate_lyrics(
                    song_name=song_name,
                    audio_path=input_path,
                    whisper_model=whisper_model
                )

                cleaned_lyrics, word_count, unique_word_count, word_freq = process_lyrics(lyrics)

                print(f"Word count after cleaning: {word_count}")
                print(f"Unique word count: {unique_word_count}")

                # -----------------------------
                # Fingerprint
                # -----------------------------
                duration, fingerprint = fingerprint_file(input_path)

                if fingerprint is None:
                    print(f"Skipping {song_name}: fingerprint failed.")
                    conn.rollback()
                    continue

                print(f"Duration: {duration}")
                print("Fingerprint created.")

                # -----------------------------
                # MFCC Audio Features
                # -----------------------------
                tempo, mfcc_values = extract_audio_features(input_path)

                if mfcc_values is None:
                    print(f"Skipping {song_name}: MFCC extraction failed.")
                    conn.rollback()
                    continue

                print(f"Tempo: {tempo}")
                print(f"MFCC values extracted: {len(mfcc_values)}")

                # -----------------------------
                # Insert Song
                # -----------------------------
                song_id = insert_song(
                    cursor=cursor,
                    title=song_name,
                    duration=duration,
                    lyrics=lyrics,
                    word_count=word_count,
                    unique_word_count=unique_word_count,
                    word_freq=word_freq,
                    cleaned_lyrics=cleaned_lyrics

                )

                # -----------------------------
                # Insert Fingerprint
                # -----------------------------
                insert_fingerprint(
                    cursor=cursor,
                    song_id=song_id,
                    fingerprint=fingerprint
                )

                # -----------------------------
                # Insert Audio Features
                # -----------------------------
                insert_audio_features(
                    cursor=cursor,
                    song_id=song_id,
                    tempo=tempo,
                    mfcc_values=mfcc_values
                )

                conn.commit()

                print(f"Finished loading: {song_name}")

            except Exception as e:
                conn.rollback()
                print(f"Failed processing {song_name}: {e}")

    finally:
        conn.close()

    print("\nDONE: Songs, lyrics, fingerprints, and MFCC audio features loaded.")


# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------

if __name__ == "__main__":
    load_songs()