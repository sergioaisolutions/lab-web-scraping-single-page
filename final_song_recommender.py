import pickle
import pandas as pd
import spotipy
import random
from random import choice
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials

cluster_0 = pd.read_csv('data/cluster_0.csv')
cluster_1 = pd.read_csv('data/cluster_1.csv')
cluster_2 = pd.read_csv('data/cluster_2.csv')
songs_titles = pd.read_csv('data/top100_songs.csv')['title'].to_list()

kmeans = pickle.load(open('models/kmeans.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

auth = SpotifyClientCredentials(client_id='9b7031633307470bafd9f8bbec7d32c1', 
                                client_secret='2c57401f088246639991c3f4bc46f36e')

spotify_wrapper = spotipy.Spotify(auth_manager = auth)

def get_song_features(user_song):

    song = spotify_wrapper.search(q = user_song, type = "track", limit = 1)
    song_uri = song['tracks']['items'][0]['uri']
    song_features = spotify_wrapper.audio_features(song_uri)[0]
    values = [list(song_features.values())]
    columns = list(song_features.keys())

    return pd.DataFrame(data = values, columns = columns)

def get_cluster(song_features):
    song_features_numerical = song_features.drop(columns = ['mode', 'time_signature']).select_dtypes(include = np.number)
    song_features_array = scaler.transform(song_features_numerical)
    return list(kmeans.predict(song_features_array))[0]

def get_recommendation():

    user_song = input('GIVE ME A SONG: ')

    if user_song in songs_titles:

        songs_titles_copy = songs_titles
        songs_titles_copy.remove(user_song)
        selected_song = random.choice(songs_titles_copy)

    else:
        selected_song = 'No recommendation'

    if selected_song != 'No recommendation':
        return f'Song recommendation: {selected_song}'

    else:

        condition = True

        while condition:

            condition = False
        
            try:
                user_song_features = get_song_features(user_song)
            except IndexError:
                condition = True
                user_song =  input('THAT SONG NOT EXIST. CHOOSE ANOTHER: ')

        user_song_cluster = get_cluster(user_song_features)

        if user_song_cluster == 0:
            titles = list(cluster_0['song'].unique())
            song = choice(titles)
            artist = cluster_0['artist'][cluster_0['song'] == song].values[0]
            print(f'Song recommendation: {song} - {artist}')

        elif user_song_cluster == 1:
            titles = list(cluster_1['song'].unique())
            song = choice(titles)
            artist = cluster_1['artist'][cluster_1['song'] == song].values[0]
            print(f'Song recommendation: {song} - {artist}')

        else:
            titles = list(cluster_2['song'].unique())
            song = choice(titles)
            artist = cluster_2['artist'][cluster_2['song'] == song].values[0]
            print(f'Song recommendation: {song} - {artist}')

get_recommendation()