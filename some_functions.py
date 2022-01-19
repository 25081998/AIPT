import numpy as np
import pandas as pd
import dask.array as da

import json
import pickle
import warnings
import random
import zipfile 
import os
import collections
import datetime
import re
import scipy
import sys

warnings.filterwarnings('ignore')

class Stats:

    def __init__(self, quick=False, max_files_for_quick_processing=5):
        self.total_playlists = 0
        self.total_tracks = 0
        self.tracks = set()
        self.artists = set()
        self.albums = set()
        self.titles = set()
        self.total_descriptions = 0
        self.ntitles = set()
        self.title_histogram = collections.Counter()
        self.artist_histogram = collections.Counter()
        self.track_histogram = collections.Counter()
        self.last_modified_histogram = collections.Counter()
        self.num_edits_histogram = collections.Counter()
        self.playlist_length_histogram = collections.Counter()
        self.num_followers_histogram = collections.Counter()

        self.quick = quick
        self.max_files_for_quick_processing = max_files_for_quick_processing


    def __getstate__(self) -> dict:
        state = {}
        
        state["total_playlists"] = self.total_playlists
        state["total_tracks"] = self.total_tracks
        state["tracks"] = self.tracks
        state["artists"] = self.artists
        state["albums"] = self.albums
        state["titles"] = self.titles
        state["total_descriptions"] = self.total_descriptions
        state["ntitles"] = self.ntitles
        state["title_histogram"] = self.title_histogram
        state["artist_histogram"] = self.artist_histogram
        state["track_histogram"] = self.track_histogram
        state["last_modified_histogram"] = self.last_modified_histogram
        state["num_edits_histogram"] = self.num_edits_histogram
        state["playlist_length_histogram"] = self.playlist_length_histogram
        state["num_followers_histogram"] = self.num_followers_histogram
      
        state["quick"] = self.quick
        state["max_files_for_quick_processing"] = self.max_files_for_quick_processing

        return state
    
    # для загрузки объекта
    def __setstate__(self, state: dict):
        self.total_playlists = state["total_playlists"]
        self.total_tracks = state["total_tracks"]
        self.tracks = state["tracks"]
        self.artists = state["artists"]
        self.albums = state["albums"]
        self.titles = state["titles"]
        self.total_descriptions = state["total_descriptions"]
        self.ntitles = state["ntitles"]
        self.title_histogram = state["title_histogram"]
        self.artist_histogram = state["artist_histogram"]
        self.track_histogram = state["track_histogram"]
        self.last_modified_histogram = state["last_modified_histogram"]
        self.num_edits_histogram = state["num_edits_histogram"]
        self.playlist_length_histogram = state["playlist_length_histogram"]
        self.num_followers_histogram = state["num_followers_histogram"]

        self.quick = state["quick"]
        self.max_files_for_quick_processing = state["max_files_for_quick_processing"]

    def process_mpd(self, path, summary=False):
        count = 0

        with zipfile.ZipFile(path, "r") as z:
            for file_info in z.infolist():
                if ('.json' in file_info.filename) & ('mpd.slice.' in file_info.filename):
                    with z.open(file_info.filename) as f:  
                        js = f.read() 
                        mpd_slice = json.loads(js)

                        self.process_info(mpd_slice["info"])

                        for playlist in mpd_slice["playlists"]:
                            self.process_playlist(playlist)

                count += 1
                if self.quick and count > self.max_files_for_quick_processing:
                      break

        if summary:
            self.show_summary()

    def show_summary(self, k=20):
        print()
        print("number of playlists", self.total_playlists)
        print("number of tracks", self.total_tracks)
        print("number of unique tracks", len(self.tracks))
        print("number of unique albums", len(self.albums))
        print("number of unique artists", len(self.artists))
        print("number of unique titles", len(self.titles))
        print("number of playlists with descriptions", self.total_descriptions)
        print("number of unique normalized titles", len(self.ntitles))
        print("avg playlist length", float(self.total_tracks) / self.total_playlists)
        print()
        print("top playlist titles")
        for title, count in self.title_histogram.most_common(k):
            print("%7d %s" % (count, title))

        print()
        print("top tracks")
        for track, count in self.track_histogram.most_common(k):
            print("%7d %s" % (count, track))

        print()
        print("top artists")
        for artist, count in self.artist_histogram.most_common(k):
            print("%7d %s" % (count, artist))

        print()
        print("numedits histogram")
        for num_edits, count in self.num_edits_histogram.most_common(k):
            print("%7d %d" % (count, num_edits))

        print()
        print("last modified histogram")
        for ts, count in self.last_modified_histogram.most_common(k):
            print("%7d %s" % (count, Stats.to_date(ts)))

        print()
        print("playlist length histogram")
        for length, count in self.playlist_length_histogram.most_common(k):
            print("%7d %d" % (count, length))

        print()
        print("num followers histogram")
        for followers, count in self.num_followers_histogram.most_common(k):
            print("%7d %d" % (count, followers))

    def normalize_name(name):
        name = name.lower()
        name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name

    def to_date(epoch):
        return datetime.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d")

    def process_playlist(self, playlist):
        self.total_playlists += 1
        # print playlist['playlist_id'], playlist['name']

        if "description" in playlist:
            self.total_descriptions += 1

        self.titles.add(playlist["name"])
        nname = Stats.normalize_name(playlist["name"])
        self.ntitles.add(nname)
        self.title_histogram[nname] += 1

        self.playlist_length_histogram[playlist["num_tracks"]] += 1
        self.last_modified_histogram[playlist["modified_at"]] += 1
        self.num_edits_histogram[playlist["num_edits"]] += 1
        self.num_followers_histogram[playlist["num_followers"]] += 1

        for track in playlist["tracks"]:
            self.total_tracks += 1
            self.albums.add(track["album_uri"])
            self.tracks.add(track["track_uri"])
            self.artists.add(track["artist_uri"])

            full_name = track["track_name"] + " by " + track["artist_name"]
            self.artist_histogram[track["artist_name"]] += 1
            self.track_histogram[full_name] += 1  

    def process_info(self, _):
        pass 

def read_slice(path, slice_):
    with zipfile.ZipFile(path, "r") as z:
      #for file_info in z.infolist():
        try:
            with z.open(slice_) as f:  
                js = f.read() 
                return json.loads(js)
        except:
            print('Slice name error')
            
def make_playlist_df(slice_):
    return pd.DataFrame.from_dict(slice_['playlists'], orient='columns')
            
def make_track_df(slice_, cols):
    df_slice = make_playlist_df(slice_)
    songs = []

    for index, row in df_slice.iterrows():
        for track in row['tracks']:
            arr = [row['pid']]
            for col in cols:
                arr.append(track[col])
            songs.append(arr)

    return pd.DataFrame(songs, columns=['pid'] + cols)

def make_DataFrame(track_js):
    return pd.DataFrame.from_dict(json.loads(track_js), orient='index')

def make_json(track_dict):
    return json.dumps(track_dict)

def make_data_base(path):
    track_dict = {}
    
    with zipfile.ZipFile(path, "r") as z:
        for file_info in z.infolist():
            if ('.json' in file_info.filename) & ('mpd.slice.' in file_info.filename):
                with z.open(file_info.filename) as f:  
                    js = f.read() 
                    mpd_slice = json.loads(js)

                    for playlist in mpd_slice["playlists"]:
                        for track in playlist['tracks']:
                            if track['track_uri'] not in track_dict.keys():
                                track_dict[track['track_uri']] = {'track_name': track['track_name'], 
                                                                  'artist_name': track['artist_name'],
                                                                  'album_name': track['album_name'],
                                                                  'duration_ms': track['duration_ms']}
    return json.dumps(track_dict)