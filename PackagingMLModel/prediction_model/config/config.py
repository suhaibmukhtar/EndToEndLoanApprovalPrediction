import os
from datetime import datetime
from pathlib import Path

SUB_PACKAGE = Path(os.path.abspath(os.path.dirname(__file__))).parent
DATASET_DIR = os.path.join(SUB_PACKAGE,"datasets")
DATA_FILE_NAME = "loan_approval_dataset.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

#dataset columns to be used for model training and prediction, the target column is 'loan_status'
#dropping columns 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value' as they are already in 'total_assets_value'
#dropping 'loan_id' as it is not needed
FEATURES = ['no_of_dependents', 'education', 'self_employed','income_annum', 'loan_amount', 'loan_term', 'cibil_score',
            'total_assets_value', 'loan_status']
TARGET = 'loan_status'

MODEL_NAME = 'loan_approval_model.pkl'
SAVE_MODEL_PATH = os.path.join(SUB_PACKAGE, 'trained_models')

ROOT_DIR = os.path.dirname(SUB_PACKAGE)
LOG_DIR = os.path.join(ROOT_DIR, 'Logs') 
LOG_FILE_NAME = f"{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

TEST_SIZE = 0.2
    
#CUSTOM COLUMN
CUSTOM_COLUMN_NAME = 'total_assets_value'
COLUMNS_TO_MERGE = ['residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']

CATEGORICAL_FEATURES_TO_ENCODE = ['education', 'self_employed']
LOG_TRANSFORMATION = ['income_annum', 'loan_amount', 'cibil_score', CUSTOM_COLUMN_NAME]

COLUMNS_TO_DROP = ['loan_id','residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']

TRAIN_CONFUSION_MATRIX_PATH = os.path.join("PackagingMLModel",SUB_PACKAGE, 'trained_models', 'confusion_matrix_Train.png')
TEST_CONFUSION_MATRIX_PATH = os.path.join("PackagingMLModel",SUB_PACKAGE, 'trained_models', 'confusion_matrix_Test.png')