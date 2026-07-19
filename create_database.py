""""create_database.py" - This script creates a SQLite database with two tables: 'songs' and 'audio_versions'."""

import sqlite3

def create_database(db_name="songs_database.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # -----------------------------
    # SONGS TABLE (main metadata)
    # -----------------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS songs (
        song_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        duration REAL,
        lyrics TEXT,
        word_count INTEGER,
        unique_word_count INTEGER,
        word_freq TEXT,
        cleaned_lyrics TEXT
    );
    """)

    # -----------------------------
    # FINGERPRINTS TABLE test
    # (handles tempo + pitch variants)
    # -----------------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fingerprints (
        fingerprint_id INTEGER PRIMARY KEY AUTOINCREMENT,
        song_id INTEGER NOT NULL,
        fingerprint TEXT NOT NULL,
        FOREIGN KEY (song_id) REFERENCES songs(song_id)
    );
    """)
      # -----------------------------
    # AUDIO FEATURES TABLE
    # (handles MFCCs and other audio features)
    # -----------------------------

    cursor.execute("""CREATE TABLE IF NOT EXISTS audio_features (
        feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
        song_id INTEGER NOT NULL,

        tempo REAL,

        mfcc_1 REAL,
        mfcc_2 REAL,
        mfcc_3 REAL,
        mfcc_4 REAL,
        mfcc_5 REAL,
        mfcc_6 REAL,
        mfcc_7 REAL,
        mfcc_8 REAL,
        mfcc_9 REAL,
        mfcc_10 REAL,
        mfcc_11 REAL,
        mfcc_12 REAL,
        mfcc_13 REAL,
        mfcc_14 REAL,
        mfcc_15 REAL,
        mfcc_16 REAL,
        mfcc_17 REAL,
        mfcc_18 REAL,
        mfcc_19 REAL,
        mfcc_20 REAL,

        FOREIGN KEY (song_id) REFERENCES songs(song_id)
    );""")
    
    conn.commit()
    conn.close()
    print("Database and tables created successfully.")

