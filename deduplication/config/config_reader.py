import os
import yaml

BASE_PATH = os.path.dirname(__file__)
filepath = os.path.join(BASE_PATH, f'config.yml')

with open(filepath, "r") as f:
    config = yaml.safe_load(f)

