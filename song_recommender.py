import spotipy
import numpy as np
import requests as re
from random import choice
from bs4 import BeautifulSoup
from IPython.display import display
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import random

page = re.get('https://www.billboard.com/charts/hot-100')
soup = BeautifulSoup(page.content, 'html.parser')

songs_titles = list()
songs_artists = list()

for song_title in soup.select('span.chart-element__information__song'):
    songs_titles.append(song_title.get_text())

for song_artist in soup.select('span.chart-element__information__artist'):
    songs_artists.append(song_artist.get_text())

songs = pd.DataFrame({'title': songs_titles, 'artist': songs_artists})

new_urls = ['https://www.elportaldemusica.es/lists/top-100-canciones/2018', 
            'https://www.elportaldemusica.es/lists/top-100-canciones/2019',
            'https://www.elportaldemusica.es/lists/top-100-canciones/2020']

songs_titles_2 = list()
songs_artists_2 = list()

for url in new_urls:
    
    page_2 = re.get(url)
    soup_2 = BeautifulSoup(page_2.text, 'html.parser')

    for song_title in soup_2.select('div.name'):

        song_title = song_title.get_text()
        song_title = song_title.strip()
        song_title = song_title.rstrip("\n")
        songs_titles_2.append(song_title)

    for song_artist in soup_2.select('div.related'):

        song_artist = song_artist.get_text()
        song_artist = song_artist.strip()
        song_artist = song_artist.rstrip("\n")
        songs_artists_2.append(song_artist)

new_songs = pd.DataFrame({'title':songs_titles_2, 'artist':songs_artists_2})
total_songs = pd.concat([songs, new_songs])
total_songs = total_songs.drop_duplicates(subset = 'title')

auth = SpotifyClientCredentials(client_id='9b7031633307470bafd9f8bbec7d32c1', 
                                client_secret='2c57401f088246639991c3f4bc46f36e')
                                
spotify_wrapper = spotipy.Spotify(auth_manager = auth)

spotify_songs_data = pd.read_csv('spotify_songs_data.csv').dropna()

spotify_songs_data_numerical = spotify_songs_data.drop(columns = ['mode', 'time_signature']).select_dtypes(include = np.number).dropna()
scaler = StandardScaler().fit(spotify_songs_data_numerical)
spotify_songs_data_numerical_array = scaler.transform(spotify_songs_data_numerical)
spotify_songs_data_numerical_scaled = pd.DataFrame(spotify_songs_data_numerical_array, columns = spotify_songs_data_numerical.columns)

kmeans = KMeans(n_clusters = 4, random_state = 1234)
kmeans.fit(spotify_songs_data_numerical_scaled)
spotify_songs_data['cluster'] = kmeans.predict(spotify_songs_data_numerical_scaled)

cluster_0 = spotify_songs_data[spotify_songs_data['cluster'] == 0]
cluster_1 = spotify_songs_data[spotify_songs_data['cluster'] == 1]
cluster_2 = spotify_songs_data[spotify_songs_data['cluster'] == 2]

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

    print('\nSONG RECOMMENDER')
    print('----------------\n')

    user_song = input('- GIVE ME A SONG: ')

    if user_song in songs_titles:

        songs_titles_copy = songs_titles
        songs_titles_copy.remove(user_song)
        selected_song = random.choice(songs_titles_copy)

    else:
        selected_song = 'No recommendation'

    if selected_song != 'No recommendation':
        return f'SONG RECOMMENDATION: {selected_song}\n'

    else:
        
        try:
            user_song_features = get_song_features(user_song)
        except IndexError:
            return 'ERROR! THAT SONG NOT EXIST :(\n'

        user_song_cluster = get_cluster(user_song_features)

        if user_song_cluster == 0:
            titles = list(cluster_0['song'].unique())
            return f'SONG RECOMMENDATION: {choice(titles)}\n'

        elif user_song_cluster == 1:
            titles = list(cluster_1['song'].unique())
            return f'SONG RECOMMENDATION: {choice(titles)}\n'

        else:
            titles = list(cluster_2['song'].unique())
            return f'SONG RECOMMENDATION: {choice(titles)}\n'

print(get_recommendation())