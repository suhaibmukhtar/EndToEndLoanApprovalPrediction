import os
import sys
import pandas as pd
from pathlib import Path
import mlflow


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("END-TO-END-LOAN-APPROVAL-PRediction")

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from logger import logging
from exception import CustomException
import prediction_model.processing.data_preprocessing as dp
from prediction_model.processing.data_handling import load_dataset, separate_data, data_split_strategy, save_pipeline, load_pipeline
import joblib
import prediction_model.pipeline as pipe

def run_training_pipeline():
    try:
        mlflow.sklearn.autolog()
        
        with mlflow.start_run() as run:
            logging.info("Starting Data Handling")
            #loading the dataset
            df = load_dataset(config.DATA_FILE_NAME)
            logging.info("Dataset loaded successfully")
            logging.info(f"Dataset shape:{df.shape}")
            logging.info(f"Columns in the dataset:{df.columns}")
            logging.info(f"Columns with missing values:{df.isna().sum()}")
            logging.info(f"duplicate rows in the dataset:{df.duplicated().sum()}")
            X, y = separate_data(df)
            logging.info(f"Nan-values in y:{y.isna().sum()}")
            logging.info(f"Target variable distribution:{y.value_counts()}")
            logging.info("Data separated successfully")
            X = dp.CreateCustomColumns(X)
            logging.info("Custom column created successfully")
            X = dp.ColumnsToDrop(X)
            logging.info("Columns dropped successfully")
            X= dp.TransformingNumericFeatures(X)
            y = dp.EncodingTargetVariable(y)
            logging.info("Target variable encoded successfully")
            X_train, X_test, y_train, y_test = data_split_strategy(X, y)
            logging.info("Data split successfully")
            logging.info(f"Training set shape:{X_train.shape}")
            logging.info(f"Testing set shape:{X_test.shape}")
            pd.concat([X_train,y_train],axis=1).to_csv(os.path.join(PACKAGE_ROOT,config.DATASET_DIR,config.TRAIN_FILE_NAME), index=False, header=True)
            pd.concat([X_test,y_test],axis=1).to_csv(os.path.join(PACKAGE_ROOT,config.DATASET_DIR,config.TEST_FILE_NAME), index=False, header=True)
            logging.info("Training and Testing set saved successfully")
            logging.info("Starting Data Preprocessing")        
            # Log parameters
            mlflow.log_param("data_file_name", config.DATA_FILE_NAME)
            mlflow.log_param("train_file_name", config.TRAIN_FILE_NAME)
            mlflow.log_param("test_file_name", config.TEST_FILE_NAME)

            #pipeline
            pipe.classification_pipeline.fit(X_train, y_train)
            logging.info("Pipeline fit successfully")
            save_pipeline(pipe.classification_pipeline)
            logging.info("Model saved successfully")
            mlflow.sklearn.log_model(pipe.classification_pipeline, "model")
            logging.info("Model logged successfully")
            mlflow.log_artifact(os.path.join(PACKAGE_ROOT,config.DATASET_DIR,config.TRAIN_FILE_NAME))
            mlflow.log_artifact(os.path.join(PACKAGE_ROOT,config.DATASET_DIR,config.TEST_FILE_NAME))
            logging.info("Artifacts logged successfully")
            mlflow.log_params(config.__dict__)
            logging.info("Parameters logged successfully")
            mlflow.log_artifact(__file__)
            logging.info("Code logged successfully")
    except Exception as e:
        raise CustomException(e,sys)

if __name__ == "__main__":
    run_training_pipeline()
