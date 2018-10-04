import json

with open('../utils/config.json', 'r') as f:
    config = json.load(f)


def get_config(top_level_config_name):
    return config[top_level_config_name]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
