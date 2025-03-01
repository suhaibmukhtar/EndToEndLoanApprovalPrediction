import logging
import os
from prediction_model.config import config
from pathlib import Path

if not os.path.exists(config.LOG_DIR):
    os.makedirs(config.LOG_DIR)

# Ensure the directory for the log file exists
log_file_path = Path(config.LOG_FILE_PATH)
log_file_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(lineno)d:%(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)

if __name__ == '__main__':
    logging.info("Starting Logging Information")