from utils.io_utils import json_loader, yaml_loader
from utils.spotify_utils import get_client, get_songs_and_features_dataframe

if __name__ == "__main__":
    config = yaml_loader("./config/config.yml")
    credentials = json_loader(config["credentials_path"])

    for user_name, user_id in config["users"].items():
        print("Getting {} songs".format(user_name))
        client = get_client(user_id, credentials)
        current_songs = get_songs_and_features_dataframe(user_id, credentials)
        current_songs["LABEL"] = user_name
        current_songs.to_parquet("{}_songs.parquet".format(user_name))
        print("Songs saved on {}_songs.parquet".format(user_name))