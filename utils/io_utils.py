import json
import yaml


def json_loader(json_path):
    with open(json_path, "r") as f:
        loaded_json = json.load(f)
    return loaded_json


def yaml_loader(yaml_path):
    with open(yaml_path, "r") as f:
        loaded_yaml = yaml.safe_load(f)
    return loaded_yaml
