from utils.io_utils import json_loader, yaml_loader, check_if_exists
from utils.spotify_utils import get_client, get_songs_and_features_dataframe

if __name__ == "__main__":
    config = yaml_loader("./config/config.yml")
    credentials = json_loader(config["credentials_path"])
    check_if_exists(config["data_path"], create=True)
    for user_name, user_id in config["users"].items():
        print("Getting {} songs".format(user_name))
        client = get_client(user_id, credentials)
        current_songs = get_songs_and_features_dataframe(user_id, credentials)
        current_songs["LABEL"] = user_name
        current_path = f"{config['data_path']}/{user_name}_songs.parquet"
        current_songs.to_parquet(current_path, index=False)
        print("Songs saved on {}_songs.parquet".format(user_name))
