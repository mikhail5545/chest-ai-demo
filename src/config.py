import json

with open(file="/mnt/c/Users/mikai/chest-ai/src/config.json", mode='r', encoding="UTF-8") as conf_file:
    conf_dict = json.load(conf_file)


MIN_CASES = int(conf_dict["min-cases"])
CHECKPOINT_WEIGHTS_PATH = conf_dict["model-properties"][0]["checkpoint-weights-path"]
WEIGHTS_PATH = conf_dict["model-properties"][0]["weights-path"]
COLOR_MODE = conf_dict["model-properties"][0]["color-mode"]
CLASS_MODE = conf_dict["model-properties"][0]["class-mode"]
PREDICITONS_DIRECTORY = conf_dict["predictions-directory"]