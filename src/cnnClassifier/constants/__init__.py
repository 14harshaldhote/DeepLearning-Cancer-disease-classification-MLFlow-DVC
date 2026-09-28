from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

# Keras' flow_from_directory sorts class folders alphabetically.
CLASS_NAMES = ["adenocarcinoma", "normal"]
CLASS_LABELS = {"adenocarcinoma": "Adenocarcinoma Cancer", "normal": "Normal"}
