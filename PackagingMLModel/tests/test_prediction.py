import pytest
import os
import sys
from pathlib import Path
import pandas as pd

import mlflow
from logger import logging
from exception import CustomException
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset, separate_data,load_pipeline, data_split_strategy
from prediction_model.processing.data_preprocessing import CreateCustomColumns, ColumnsToDrop, EncodingTarget
from prediction_model.prediction_pipeline import generate_predictions


def test_load_pipeline():
    try:
        classification_pipeline = load_pipeline(config.MODEL_NAME)
        # Ensure the pipeline loads successfully
        assert classification_pipeline is not None, "Pipeline failed to load"
        # Ensure it's an instance of a model pipeline (or check expected attributes)
        assert hasattr(classification_pipeline, "predict"), "Loaded object is not a valid model pipeline"
    except CustomException:
        pytest.fail("CustomException raised: Model loading failed")
    except FileNotFoundError:
        pytest.fail("FileNotFoundError: Model file not found")
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")
        
def test_separate_data():
    try:
        df = load_dataset(config.DATA_FILE_NAME)
        X, y = separate_data(df)
        # Ensure the data is separated correctly
        assert isinstance(X, pd.DataFrame), "X is not a DataFrame"
        assert isinstance(y, pd.Series), "y is not a Series"
        # Ensure the target variable is in the y variable
        assert config.TARGET in df.columns, "Target variable not in Dataset"
        assert len(X.columns) == len(df.columns) - 1, "Data not separated correctly"
    except CustomException:
        pytest.fail("CustomException raised: Data separation failed")
    except FileNotFoundError:
        pytest.fail("FileNotFoundError: Data file not found")
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")

def test_data_split_strategy():
    try:
        df = load_dataset(config.DATA_FILE_NAME)
        X, y = separate_data(df)
        X_train, X_test, y_train, y_test = data_split_strategy(X, y)
        # Ensure the data is split correctly
        assert isinstance(X_train, pd.DataFrame), "X_train is not a DataFrame"
        assert isinstance(X_test, pd.DataFrame), "X_test is not a DataFrame"
        assert isinstance(y_train, pd.Series), "y_train is not a Series"
        assert isinstance(y_test, pd.Series), "y_test is not a Series"
        # Ensure the data is split in the correct ratio
        assert X_train.shape[0] == 3415, "Training set shape is incorrect"
        assert X_test.shape[0] == 854, "Testing set shape is incorrect"
    except CustomException:
        pytest.fail("CustomException raised: Data split failed")
    except FileNotFoundError:
        pytest.fail("FileNotFoundError: Data file not found")
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")
        
def test_CreateCustomColumns():
    try:
        df = load_dataset(config.DATA_FILE_NAME)
        X, y = separate_data(df)
        X = CreateCustomColumns(X)
        # Ensure the custom column is created correctly 
        assert config.CUSTOM_COLUMN_NAME in X.columns, "Custom column not created"
        assert len(X.columns) == len(df.columns), "Custom column not created correctly" #bcz label col is droped from X
    except CustomException:
        pytest.fail("CustomException raised: Custom column creation failed")
    except FileNotFoundError:
        pytest.fail("FileNotFoundError: Data file not found")
    except Exception as e:  
        pytest.fail(f"Unexpected error: {e}")