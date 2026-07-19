"""
fingerprints_comparison.py

This will take the fingerprint from the uploaded song and compare it to the fingerprints in the Songs_database for a match

Since it is unlikely that a variant fingerprint will match any song, only the original uploaded song will compare fingerprint.


"""


import pandas as pd
import sqlite3
import Levenshtein

# Connect to your SQLite database
conn = sqlite3.connect("songs_database.db")

# Load each table into a DataFrame
songs_df = pd.read_sql_query("SELECT song_id, title FROM songs", conn)
fingerprints_df = pd.read_sql_query("SELECT song_id,fingerprint FROM fingerprints", conn)
fingerprints_df["clean_fp"] = fingerprints_df["fingerprint"].astype(str).str.strip()

uploaded_song_conn = conn = sqlite3.connect("uploaded_song_database.db")

uploaded_song_fingerprint = pd.read_sql_query("SELECT fingerprint FROM fingerprints", conn)


# To compare the fingerprints of the song variants change the location to 0-4 for the original and 4 variant versions of the song
#  0 = original uploaded song , 1-4 are variants.

location = 0

uploaded_fingerprint = uploaded_song_fingerprint["fingerprint"].iloc[location].strip()

match = fingerprints_df[fingerprints_df["clean_fp"] == uploaded_fingerprint]


fingerprints_df["similarity"] = fingerprints_df["clean_fp"].apply(
    lambda fp: Levenshtein.ratio(fp, uploaded_fingerprint)
)

best_match = fingerprints_df.sort_values("similarity", ascending=False).iloc[0]
print(best_match)
print(best_match.song_id)