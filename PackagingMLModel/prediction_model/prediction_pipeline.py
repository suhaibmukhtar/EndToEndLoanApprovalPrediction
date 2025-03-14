import os
import sys
import pandas as pd
from pathlib import Path
import numpy as np
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

mlflow.set_tracking_uri("http://localhost:5000") #default path to mlartifact
mlflow.set_experiment("END-TO-END-LOAN-APPROVAL-PRediction") #experiment name

from prediction_model.config import config
from logger import logging
from exception import CustomException
import prediction_model.processing.data_preprocessing as dp
from prediction_model.processing.data_handling import load_dataset, separate_data, data_split_strategy, save_pipeline, load_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    classification_pipeline = load_pipeline(config.MODEL_NAME)
    logging.info("Model loaded successfully")
    # preprocessor = classification_pipeline.named_steps['preprocessor'] to obtain only preprocessor from pipeline
except Exception as e:
    raise CustomException(e,sys)

def generate_predictions():
    try:
        mlflow.sklearn.autolog()
        with mlflow.start_run(run_name='Evaluation-Loan_prediction') as run:
            mlflow.set_tag("mlflow.user", "Suhaib_Mukhtar")
            test_data = load_dataset(config.TEST_FILE_NAME)
            train_data = load_dataset(config.TRAIN_FILE_NAME)
            mlflow.data.from_pandas(
                train_data,
                source=os.path.join(PACKAGE_ROOT,config.DATASET_DIR,config.TRAIN_FILE_NAME),
                name="train_data",
            )
            mlflow.data.from_pandas(
                test_data,
                source=os.path.join(PACKAGE_ROOT,config.DATASET_DIR,config.TEST_FILE_NAME),
                name="test_data",
            )
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

def EvaluationMetrics(y_pred_class, y_test, subset):
    try:
        accuracy = accuracy_score(y_test, y_pred_class)
        precision = precision_score(y_test, y_pred_class)
        recall = recall_score(y_test, y_pred_class)
        f1 = f1_score(y_test, y_pred_class)
        cm = confusion_matrix(y_test, y_pred_class)
        logging.info(f"Confusion Matrix:{cm}")
        logging.info(f"Accuracy:{accuracy}")
        logging.info(f"Precision:{precision}")
        logging.info(f"Recall:{recall}")
        logging.info(f"F1 Score:{f1}")
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix_{subset}')
        
        # Ensure the directory exists
        confusion_matrix_path = os.path.join(PACKAGE_ROOT, config.SAVE_MODEL_PATH, f"confusion_matrix_{subset}.png")        
        plt.savefig(confusion_matrix_path)
        
        if subset == "Test":
            mlflow.log_metric("Test_Accuracy", accuracy)
            mlflow.log_metric("Test_Precision", precision)
            mlflow.log_metric("Test_Recall", recall)
            mlflow.log_metric("Test_F1_Score", f1)
            mlflow.log_artifact(confusion_matrix_path)
        else:
            mlflow.log_metric("Train_Accuracy", accuracy)
            mlflow.log_metric("Train_Precision", precision)
            mlflow.log_metric("Train_Recall", recall)
            mlflow.log_metric("Train_F1_Score", f1)
            mlflow.log_artifact(confusion_matrix_path)
            
        mlflow.log_artifact(__file__) #log the current file
        return {"Accuracy":accuracy, "Precision":precision, "Recall":recall, "f1-score":f1}
    except Exception as e:
        raise CustomException(e,sys)

if __name__=="__main__":
    y_pred_class, y_test = generate_predictions()
    y_pred_train_class, y_train = generate_predictions()
    results = EvaluationMetrics(y_pred_class, y_test, subset="Test")
    train_results = EvaluationMetrics(y_pred_train_class, y_train, subset="Train")
    logging.info("Evaluation Metrics calculated successfully")
    
