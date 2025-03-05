import os
import sys
import pandas as pd
from pathlib import Path
import numpy as np

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from logger import logging
from exception import CustomException
import prediction_model.processing.data_preprocessing as dp
from prediction_model.processing.data_handling import load_dataset, separate_data, data_split_strategy, save_pipeline, load_pipeline

try:
    classification_pipeline = load_pipeline(config.MODEL_NAME)
    logging.info("Model loaded successfully")
    # preprocessor = classification_pipeline.named_steps['preprocessor'] to obtain only preprocessor from pipeline
except Exception as e:
    raise CustomException(e,sys)

def generate_predictions():
    try:
        test_data = load_dataset(config.TEST_FILE_NAME)
        logging.info("Test Data loaded successfully")
        X_test, y_test = separate_data(test_data)
        logging.info("Separated test data successfully")
        y_pred = classification_pipeline.predict(X_test)
        logging.info("Predictions generated successfully")
        y_pred_class = np.where(y_pred > 0.5, 1, 0)
        logging.info("Predictions converted to classes successfully")
        return y_pred_class, y_test
        
    except Exception as e:
        raise CustomException(e,sys)

if __name__=="__main__":
    y_pred_class, y_test = generate_predictions()
    logging.info(f"Predicted classes:{y_pred_class}")
    logging.info(f"Actual classes:{y_test}")