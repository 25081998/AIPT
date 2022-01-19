import numpy as np
import pandas as pd

import json
import pickle
import random
import re
import scipy

from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
from catboost import CatBoostRanker, Pool
from scipy.spatial.distance import *

random.seed(42)

def load_rank_model():
    with open('rank_model/catboostRanker_model.pkl', 'rb') as model:
        return pickle.loads(model.read())

def load_text_models():
    sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    artist_model = Word2Vec.load("w2v_model/word2vec.model")
    
    return sentence_model, artist_model

def load_similarity():
    return [cosine, euclidean]

def weight_1(track):
    return 1

def weight_2(track):
    return 1 / (track['pos'] + 1)

def name_embeddings(playlist, model):
    return model.encode(playlist['name'])

def sentense_embeddings(track, model):
    name = model.encode(track['track_name'])
    album = model.encode(track['album_name'])
    
    return {'pos': track['pos'],
            'artist_name': track['artist_name'],
            'track_uri': track['track_uri'],
            'track_emb': list(name),
            'duration_ms': int(track['duration_ms']),
            'album_emb': list(album)}

def artist_embeddings(track, model):
    artist = model.wv[track['artist_name']]
    
    return {'pos': track['pos'],
            'artist_emb': list(artist),
            'track_uri': track['track_uri'],
            'track_emb': track['track_emb'],
            'duration_ms': int(track['duration_ms']),
            'album_emb': track['album_emb']}

def playlist_pipline(playlist, sentence_model, artist_model, weight):
    durations = []
    
    for i, track in enumerate(playlist['tracks']):
        track_ = sentense_embeddings(track, sentence_model)
        track_ = artist_embeddings(track_, artist_model)
        
        if i == 0:
            average_name = np.array(track_['track_emb'])
            average_album = np.array(track_['album_emb'])
            average_artist = np.array(track_['artist_emb'])
        else:
            average_name += weight(track_) * np.array(track_['track_emb'])
            average_album += weight(track_) * np.array(track_['album_emb'])
            average_artist += weight(track_) * np.array(track_['artist_emb'])  
            
        durations.append(track_['duration_ms'])
    
    return {'pid': playlist['pid'],
            'name': name_embeddings(playlist, sentence_model),
            'max_duration': max(durations),
            'min_duration': min(durations),
            'mean_duration': sum(durations) / len(durations),
            'average_name': list(average_name / len(durations)),
            'average_album': list(average_album / len(durations)),
            'average_artist': list(average_artist / len(durations))}

def top_artists_df(vec, data, artist_model, top):
    artists = np.array(artist_model.wv.most_similar(np.array(vec), topn=top))[:, :1]
    artists = np.resize(artists, artists.shape[0]).tolist()
    df = data[data['artist_name'].apply(lambda x: x in artists)]
    
    return df

def transform_text_data(df, sentence_model, artist_model):
    d = df.copy()
    d['track_name'] = d['track_name'].apply(sentence_model.encode)
    d['album_name'] = d['album_name'].apply(sentence_model.encode)
    d['artist_name'] = d['artist_name'].apply(lambda x: artist_model.wv[x])
    
    return d

def data_pipline(playlist, data, sentence_model, artist_model, weight, similarity, top):
    playlist_info = playlist_pipline(playlist, sentence_model, artist_model, weight)
    df = top_artists_df(playlist_info['average_artist'], data, artist_model, top)
    df = transform_text_data(df, sentence_model, artist_model)
    
    for col, info in playlist_info.items():
        df[col] = [info] * df.shape[0]
        
    df['max_duration_diff'] = df['max_duration'] - df['duration_ms']
    df['min_duration_diff'] = df['min_duration'] - df['duration_ms']
    df['mean_duration_diff'] = df['mean_duration'] - df['duration_ms']
    
    for s in similarity:
        df['name_track_' + s.__name__] = df[['name', 'track_name']].apply(tuple, axis=1).apply(lambda x: s(x[0], x[1]))
        df['name_album_' + s.__name__] = df[['name', 'album_name']].apply(tuple, axis=1).apply(lambda x: s(x[0], x[1]))
        df['tracks_' + s.__name__] = df[['average_name', 'track_name']].apply(tuple, axis=1).apply(lambda x: s(x[0], x[1]))
        df['albums_' + s.__name__] = df[['average_album', 'album_name']].apply(tuple, axis=1).apply(lambda x: s(x[0], x[1]))
        df['artists_' + s.__name__] = df[['average_artist', 'artist_name']].apply(tuple, axis=1).apply(lambda x: s(x[0], x[1]))
    
    return df.drop(['name', 'max_duration', 'min_duration', 
                    'mean_duration', 'average_name', 'average_album', 
                    'average_artist', 'artist_name', 'track_name', 
                    'album_name', 'duration_ms'], axis=1)

def data_pool(df):
    X, pid = df.drop(['pid'], axis=1), df[['pid']]
    return Pool(data=X, group_id=pid)

def predict(pool, model):
    return model.predict(pool)

def rank(df, model, k):
    pool = data_pool(df)
    preds = predict(pool, model)
    df['rels'] = preds
    
    return df.sort_values(by=['pid', 'rels'], ascending=False)[['pid', 'rels']].head(k)