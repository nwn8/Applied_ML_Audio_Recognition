
# Author

Nathan S.

# Title

Applied Machine Learning Audio Recognition

### Github Repo

[https://github.com/nwn8/Applied_ML_Audio_Recognition]

# Project Instructions


###  Reproduction and local setup

1.  Clone the repository at [https://github.com/nwn8/Applied_ML_Audio_Recognition]

2.  Follow the setup process in requirements.txt to create a virtual environment and install packages

* note that some packages like openai-whisper and chromaprint require extra steps on your device outlined in requirements.txt

### Creating Source Data

3. Add 25 or more MP3 audio files to the "songs" folder.  This will be the foundation of the source data.

4.  run "py create_database.py" in the terminal  ---  This creates the database for storing the data with the following schema

SONGS TABLE

| Column | Type | Description |
| --- | --- | --- |
| **song_id** | INTEGER (PK, auto‑increment) | Unique identifier for each song |
| **title** | TEXT | Song title |
| **duration** | REAL | Duration of the track (in seconds) |
| **lyrics** | TEXT | Raw lyrics text |
| **cleaned_lyrics** | TEXT | Preprocessed lyrics (lowercased, punctuation removed, etc.) |
| **word_count** | INTEGER | Total number of words in the lyrics |
| **unique_word_count** | INTEGER | Number of unique words |
| **word_freq** | TEXT / JSON | Serialized word‑frequency dictionary |

FINGERPRINTS TABLE

| Column | Type | Description |
| --- | --- | --- |
| **fingerprint_id** | INTEGER (PK, auto‑increment) | Unique fingerprint record ID |
| **song_id** | INTEGER (FK → Songs.song_id) | Song associated with this fingerprint |
| **fingerprint** | TEXT | Chromaprint fingerprint string |

AUDIO FEATURES TABLE

| Column | Type | Description |
| --- | --- | --- |
| **feature_id** | INTEGER (PK, auto‑increment) | Unique feature record ID |
| **song_id** | INTEGER (FK → Songs.song_id) | Song associated with these features |
| **tempo** | REAL | Estimated tempo (BPM) |
| **mfcc_1** | REAL | MFCC coefficient #1 |
| **mfcc_2** | REAL | MFCC coefficient #2 |
| **…** | REAL | MFCC coefficients 3–19 |
| **mfcc_20** | REAL | MFCC coefficient #20 |


5.  Run "py loader.py" in the terminal  -  This will extract all of the audio data from the files in the "songs" folder and add the data to the songs_database.db

###  Exploratory Data Analytics

6.  Now that there is data in the database you can explore the datas various attributes and understand the data using Project_EDA.ipynb

### Extract Data from "Unknown" audio file in Uploaded_song folder

7.  Add an audio MP3 file into the uploaded_song folder and run "py uploaded_song_process.py" .  This will create an uploaded_song_database and store the same tables as the songs_database.   

###  Create the Features Dataframe and Machine Learning Model

8.  Run the notebook Project_ML_Model.ipynb to create the feature dataframe and train the model.  This notebook will also compare the uploaded "unknown" song in uploaded_song folder to the dataframe and return the closest match using KNN Machine learning model.   


