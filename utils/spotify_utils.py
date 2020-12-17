import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from tqdm import tqdm


def get_client(user, credentials):
    authenticator = SpotifyOAuth(
        client_id=credentials["CLIENT_ID"],
        client_secret=credentials["CLIENT_SECRET"],
        redirect_uri=credentials["REDIRECT_URI"],
        scope=credentials["SCOPE"],
        username=user,
    )
    token = authenticator.get_access_token(as_dict=False)
    sp = spotipy.Spotify(auth=token)
    return sp


def get_songs(client):
    songs = client.current_user_saved_tracks()
    songs_list = []
    total_songs = songs["total"]
    batch = songs["limit"]
    iterations = int(np.ceil(total_songs / batch))
    for i in tqdm(range(iterations), desc="Getting songs"):
        for song in songs["items"]:
            current_song = song["track"]
            songs_list.append(
                {
                    "name": current_song["name"],
                    "artist": [w["name"] for w in current_song["artists"]],
                    "popularity": current_song["popularity"],
                    "id": current_song["id"],
                }
            )
        songs = client.next(songs)
    songs_dataframe = pd.DataFrame(songs_list)
    return songs_dataframe


def get_features(client, songs_dataframe):
    indexes = [w for w in range(0, len(songs_dataframe), 50)]
    features = []
    for i in tqdm(indexes, desc="Getting features"):
        features = features + client.audio_features(
            songs_dataframe["id"][i : i + 50].to_list()
        )
    features = pd.DataFrame(features)
    features_dataframe = songs_dataframe.merge(features, on="id")
    return features_dataframe


def get_analysis(client, songs_dataframe):
    analysis = []
    errors = []
    for track_id in tqdm(songs_dataframe["id"], desc="Getting audio analysis"):
        try:
            current_analysis = client.audio_analysis(track_id)
            current_analysis["id"] = track_id
            analysis = analysis + [current_analysis]
        except IOError:
            errors.append(track_id)

    audio_analysis = []
    for current_analysis in tqdm(analysis, desc="Aggregating audio analysis"):
        current_summary = [current_analysis["id"]]
        current_summary += np.mean(
            [w["timbre"] for w in current_analysis["segments"]], axis=0
        ).tolist()
        current_summary += np.mean(
            [w["pitches"] for w in current_analysis["segments"]], axis=0
        ).tolist()
        audio_analysis.append(current_summary)

    audio_analysis = pd.DataFrame(audio_analysis)
    columns_name = ["id"]
    columns_name = columns_name + ["timbre_{}".format(i) for i in range(12)]
    columns_name = columns_name + ["pitches_{}".format(i) for i in range(12)]
    audio_analysis.columns = columns_name
    analysis_dataframe = songs_dataframe.merge(audio_analysis, on="id", how="inner")
    return analysis_dataframe


def get_songs_and_features_dataframe(user, credentials):
    client = get_client(user, credentials)
    songs = get_songs(client)
    features_df = get_features(client, songs)
    audio_analysis_df = get_analysis(client, features_df)
    return audio_analysis_df
