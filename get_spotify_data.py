import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
from sklearn.preprocessing import StandardScaler
from credentials import credentials_dict


CLIENT_ID = credentials_dict["CLIENT_ID"]
CLIENT_SECRET = credentials_dict["CLIENT_SECRET"]
CLIENT_URI = credentials_dict["CLIENT_URI"]
USER_1 = credentials_dict["USER_1"]
USER_2 = credentials_dict["USER_2"]
SCOPE = credentials_dict["SCOPE"]
CREDENTIALS_MANAGER = SpotifyClientCredentials(CLIENT_ID, CLIENT_SECRET)

def get_user_token(user_id, scope, client_id, client_secret, client_uri):
    token = util.prompt_for_user_token(user_id, scope, client_id, client_secret, client_uri)
    return token

def get_client(CREDENTIALS_MANAGER, token):
    client = spotipy.Spotify(client_credentials_manager=CREDENTIALS_MANAGER, auth=token)
    return client

def get_songs(client):
    songs = client.current_user_saved_tracks()
    songs_list = []
    while songs["next"]:
        for i, song in enumerate(songs['items']):
            current_song = song["track"]
            songs_list.append({"index": songs["offset"] + i,
                               "name": current_song['name'],
                               "artist": [w["name"] for w in current_song["artists"]],
                               "popularity": current_song["popularity"],
                               "id": current_song["id"]})
        songs = client.next(songs)
    songs_dataframe = pd.DataFrame(songs_list)
    return songs_dataframe

def get_features(client, songs_dataframe):
    indexes = [w for w in range(0, len(songs_dataframe), 50)]    
    features = []
    for i in indexes:
        features = features + client.audio_features(songs_dataframe["id"][i:i+50].to_list())
    features = pd.DataFrame(features)
    features_dataframe = songs_dataframe.merge(features, on="id")
    return features_dataframe

def get_analysis(client, songs_dataframe):
    analysis = []
    errors = []
    for track_id in songs_dataframe["id"]:
        try:
            current_analysis = client.audio_analysis(track_id)
            current_analysis["id"] = track_id
            analysis = analysis + [current_analysis]
        except:
            errors.append(track_id)

    audio_analysis = []
    for current_analysis in analysis:
        current_summary = [current_analysis["id"]]
        current_summary += np.mean([w["timbre"] for w in current_analysis["segments"]], axis=0).tolist()
        current_summary += np.mean([w["pitches"] for w in current_analysis["segments"]], axis=0).tolist()
        audio_analysis.append(current_summary)

    audio_analysis = pd.DataFrame(audio_analysis)
    columns_name = ["id"]
    columns_name = columns_name + ["timbre_{}".format(i) for i in range(12)]
    columns_name = columns_name + ["pitches_{}".format(i) for i in range(12)]
    audio_analysis.columns = columns_name
    analysis_dataframe = songs_dataframe.merge(audio_analysis, on="id", how="inner")
    return analysis_dataframe

def get_songs_and_features_dataframe(user_id, scope, client_id, client_secret, client_uri):
    current_token = get_user_token(user_id, scope, client_id, client_secret, client_uri)
    current_client = get_client(CREDENTIALS_MANAGER, current_token)
    current_songs = get_songs(current_client)    
    current_features_df = get_features(current_client, current_songs)
    current_audio_analysis_df = get_analysis(current_client, current_features_df)
    return current_audio_analysis_df

def normalize_features(dataframe, features):
    to_normalize = dataframe[features].copy()
    scaler = StandardScaler()
    scaled_featuers = pd.DataFrame(scaler.fit_transform(to_normalize))
    scaled_featuers.columns = to_normalize.columns
    return scaled_featuers

def get_ohe(dataframe, features):
    dummies_df = pd.concat([pd.get_dummies(data=dataframe[w], prefix=w) for w in features], axis=1)
    return dummies_df


if __name__ == "__main__":
    user_list = {USER_1: "p", 
                 USER_2: "a"}
    
    for user_name, label in user_list.items():
        print("Getting {} songs".format(user_name))
        current_songs = get_songs_and_features_dataframe(user_name, 
                                                         SCOPE, 
                                                         CLIENT_ID, 
                                                         CLIENT_SECRET, 
                                                         CLIENT_URI)
        current_songs["LABEL"] = label
        current_songs.to_parquet("{}_songs.parquet".format(label))
        print("Songs saved on {}_songs.parquet".format(label))