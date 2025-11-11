import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from tqdm import tqdm
import time
'''
    To add song titles and artist features to the music data that previously only had URIs,
    this script utilizes the Spotify API.
'''

# -----------------------------------------------------------------
# 1. Spotify API Authentication Setup
# -----------------------------------------------------------------
CLIENT_ID = ''  # Enter your Spotify API client ID
CLIENT_SECRET = ''

try:
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID,
                                            client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
except spotipy.exceptions.SpotifyException as e:
    print(f"Authentication Error: {e}")
    print("Please check if your CLIENT_ID and CLIENT_SECRET are correct.")
    exit()

train_df = pd.read_csv('../orig_data/train_set.csv')
sample_df = train_df.copy()

# Extract only the 'ID' part from the URI
uri_list = [uri.split(':')[-1] for uri in sample_df['uri']]

batch_size = 50
uri_batches = [uri_list[i:i + batch_size] for i in range(0, len(uri_list), batch_size)]

print(f"Total {len(uri_list)} URIs split into {len(uri_batches)} batches")

# -----------------------------------------------------------------
# 3. Create empty lists to store information from the API request
# -----------------------------------------------------------------
song_titles = []
album_names = []
artist_names_list = []

print("Fetching track information from Spotify API in batches...")

for batch in tqdm(uri_batches):
    try:
        # Request 50 tracks at a time using sp.tracks()
        results = sp.tracks(batch)

        # Iterate through each track result within the batch
        for track_info in results['tracks']:
            if track_info:
                # Parse information
                song_titles.append(track_info['name'])
                album_names.append(track_info['album']['name'])
                artists = [artist['name'] for artist in track_info['artists']]
                artist_names_list.append(', '.join(artists))
            else:
                # If the API returns None (e.g., for an invalid ID)
                song_titles.append(None)
                album_names.append(None)
                artist_names_list.append(None)

    except spotipy.exceptions.SpotifyException as e:
        print(f"API call error occurred: {e}. Skipping this batch.")
        # If an error occurs, fill the entire batch with None to maintain length consistency
        for _ in range(len(batch)):
            song_titles.append(None)
            album_names.append(None)
            artist_names_list.append(None)

    # 0.1-second delay to prevent hitting API Rate Limits
    time.sleep(0.1)

print("API information collection complete!")

# -----------------------------------------------------------------
# 7. Add the new information as columns to the sampled DataFrame
# -----------------------------------------------------------------
if len(sample_df) == len(song_titles):
    sample_df['track_name'] = song_titles
    sample_df['album_name'] = album_names
    sample_df['artists'] = artist_names_list
    print("Successfully added new columns to the DataFrame.")
else:
    print(f"Error: DataFrame length ({len(sample_df)}) and result list length ({len(song_titles)}) do not match!")

# -----------------------------------------------------------------
# 8. Save the final result to a CSV file
# -----------------------------------------------------------------
output_filename = '../data/train_data_last_with_metadata.csv'
sample_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print("\n--- 🎵 Final DataFrame with added info (top 5 rows) ---")
print(sample_df.head())
print(f"\n[Success] Saved to '{output_filename}'.")
