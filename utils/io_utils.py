import json
import yaml
import os
import pickle
import pandas as pd
from tensorflow.keras import models


def json_loader(json_path):
    """Loads json from a path

    Args:
        json_path (str): Path to json file

    Returns:
        dict: Dict with json content
    """
    with open(json_path, "r") as f:
        loaded_json = json.load(f)
    return loaded_json


def yaml_loader(yaml_path):
    """Loads yaml from a path

    Args:
        yaml_path (str): Path to yaml file

    Returns:
        dict: Dict with yaml content
    """
    with open(yaml_path, "r") as f:
        loaded_yaml = yaml.safe_load(f)
    return loaded_yaml


def check_if_exists(path, create=True):
    """Checks if a path exists and, if wanted, creates it

    Args:
        path (str): Path to be checked
        create (bool, optional): If path doesn't exists, creates it or not. Defaults to True.

    Returns:
        Bool: Exists or not
    """
    if os.path.exists(path):
        return True
    elif create:
        os.mkdir(path)
        return True
    else:
        return False


def get_data(config):
    """Loads data from users parquet files

    Args:
        config (dict): Configuration dictionary with user names and data path

    Returns:
        pd.DataFrame: Pandas dataframe with data from all users
    """
    data_folder = config["data_path"]
    df = pd.DataFrame()
    for user in config["users"]:
        df = pd.concat(
            [df, pd.read_parquet(f"{data_folder}/{user}_songs.parquet")]
        ).reset_index(drop=True)
    return df


def save_transformers(feature_transformer, label_encoder, config):
    """Saves transformers to a pickle file

    Args:
        feature_transformer (sklearn.ColumnTransformer): feature engineering pipeline
        label_encoder (sklearn.LabelEncoder): label encoder
        config (dict): Configuration dictionary with model path

    Returns:
        Bool: True
    """
    check_if_exists(config["model_path"], create=True)
    transformers_path = f"{config['model_path']}/transformers.pickle"
    transformer_dict = {
        "feature_transformer": feature_transformer,
        "label_encoder": label_encoder,
    }
    with open(transformers_path, "wb") as f:
        pickle.dump(transformer_dict, f)
    print(f"Transformers saved on {transformers_path}")
    return True


def load_transformers(config):
    """Function to load feature engineering and label encoder transformers

    Args:
        config (dict): Configuration dictionary with model path

    Returns:
        tuple: (feature_transformer, label_encoder)
    """
    transformers_path = f"{config['model_path']}/transformers.pickle"
    if not check_if_exists(transformers_path, create=False):
        return "Transformers not found"
    else:
        with open(transformers_path, "rb") as f:
            transformer_dict = pickle.load(f)
        print(f"Transformers loaded from {transformers_path}")
    return transformer_dict["feature_transformer"], transformer_dict["label_encoder"]


def save_model(model, config):
    """Function to save keras model

    Args:
        model (keras.model): Trained keras model
        config (dict): Configuration dictionary with model path

    Returns:
        Bool: True
    """
    check_if_exists(config["model_path"], create=True)
    model_path = f"{config['model_path']}/model.h5"
    model.save(model_path)
    print(f"Model saved on {model_path}")
    return True


def load_model(config):
    """Function to load keras model

    Args:
        config (dict): Configuration dictionary with model path

    Returns:
        keras.model: Trained keras model
    """
    model_path = f"{config['model_path']}/model.h5"
    if not check_if_exists(model_path, create=False):
        return "Model not found"
    else:
        model = models.load(model_path)
        print(f"Model saved on {model_path}")
    return model
