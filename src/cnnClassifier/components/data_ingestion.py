import shutil
import zipfile

import gdown

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.utils.common import get_size


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> None:
        """Get the dataset zip: the copy in the repo if present, otherwise Google Drive."""
        target = self.config.local_data_file
        if target.exists():
            logger.info(f"Dataset already present at {target} ({get_size(target)})")
            return

        if self.config.local_source.exists():
            shutil.copyfile(self.config.local_source, target)
            logger.info(f"Copied dataset from {self.config.local_source}")
            return

        file_id = self.config.source_URL.split("/")[-2]
        logger.info(f"Downloading dataset {file_id} from Google Drive into {target}")
        gdown.download(id=file_id, output=str(target), quiet=False)

    def extract_zip_file(self) -> None:
        self.config.unzip_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)
        logger.info(f"Extracted dataset into {self.config.unzip_dir}")
