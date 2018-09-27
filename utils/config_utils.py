import json

with open('../utils/config.json', 'r') as f:
    config = json.load(f)


def get_config(top_level_config_name):
    return config[top_level_config_name]
