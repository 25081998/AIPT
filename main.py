import eel
import numpy as np
import pandas as pd
import json
import warnings

from ranking_framework import *

warnings.filterwarnings('ignore')

print('Загрузка...')
tracks_df = pd.read_csv('data/tracks.csv').set_index('Unnamed: 0')

playlist = {'pid': 0,
            'name': '',
            'tracks': []}

position = 0

@eel.expose
def find_tracks(track_name, artist_name): 
    print('Поиск треков...')
    
    if(track_name != '') & (artist_name != ''):
        tracks_json = tracks_df[(tracks_df['track_name'].apply(str).apply(str.lower) == track_name.lower()) & (tracks_df['artist_name'].apply(str).apply(str.lower) == artist_name.lower())].to_dict(orient='index')
    elif track_name != '':
        tracks_json = tracks_df[tracks_df['track_name'].apply(str).apply(str.lower) == track_name.lower()].to_dict(orient='index')
    elif artist_name != '':
        tracks_json = tracks_df[tracks_df['artist_name'].apply(str).apply(str.lower) == artist_name.lower()].to_dict(orient='index')
    else:
        tracks_json = {}
        
    return json.dumps(tracks_json)

@eel.expose
def add_track(track): 
    print('Добавление трека в плейлист...')
    global position
    
    track_name, artist_name = track[1:-1].split(sep=' / ')
    track_json = tracks_df[(tracks_df['track_name'] == track_name) & (tracks_df['artist_name'] == artist_name)].head()
    track_json = {'pos': position,
                  'track_uri': track_json.index[0],
                  'track_name': track_json['track_name'].values[0],
                  'artist_name': track_json['artist_name'].values[0],
                  'album_name': track_json['album_name'].values[0],
                  'duration_ms': int(track_json['duration_ms'].values[0])}
    playlist['tracks'].append(track_json)
    position += 1
        
    return json.dumps(track_json)

@eel.expose
def make_recomends(name, count):
    print('Формирование рекомендаций...')
    
    if count == '':
        count = 1
    else:
        count = int(count)
    
    playlist['name'] = name

    print('...Подгрузка моделей...')
    sentence_model, artist_model = load_text_models()   
    rank_model = load_rank_model()
    similarity = load_similarity()
    
    print('...Предобработка данных...')
    data = data_pipline(playlist, tracks_df, sentence_model, artist_model, weight_1, similarity, 10)
    print('...Ранжирование рекомендаций...')
    result = rank(data, rank_model, count).merge(tracks_df, left_index=True, right_index=True, how='left')
    
    return result[['track_name', 'artist_name']].rename(columns={'track_name': 'Название трека', 'artist_name': 'Исполнитель'}).to_html(index=False)
    
@eel.expose
def clear_info():
    print('Сброс данных...')
    
    global position
    position = 0
    
    global playlist
    playlist = {'pid': 0,
                'name': '',
                'tracks': []}

eel.init('app')
eel.start('main.html', mode='chrome', size=(600, 800))
