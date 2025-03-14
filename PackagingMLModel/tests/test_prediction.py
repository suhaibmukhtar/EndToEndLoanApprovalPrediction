import pytest
import os
import sys
from pathlib import Path
import pandas as pd

from logger import logging
from exception import CustomException
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset, separate_data,load_pipeline, data_split_strategy
from prediction_model.processing.data_preprocessing import CreateCustomColumns, ColumnsToDrop, EncodingTargetVariable
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
        
def test_ColumnsToDrop():
    try:
        df = load_dataset(config.DATA_FILE_NAME)
        X, y = separate_data(df)
        X = CreateCustomColumns(X)
        X = ColumnsToDrop(X)
        # Ensure the columns are dropped correctly
        assert not any(col in X.columns for col in config.COLUMNS_TO_DROP), "Custom column(s) not dropped"
        assert len(X.columns) == (len(df.columns) - len(config.COLUMNS_TO_DROP)), "Columns not dropped correctly" #bcz label col is droped from X
    except CustomException:
        pytest.fail("CustomException raised: Columns drop failed")
    except FileNotFoundError:  
        pytest.fail("FileNotFoundError: Data file not found")
    except Exception as e:  
        pytest.fail(f"Unexpected error: {e}")
        
def test_EncodingTargetVariable():
    try:
        df = load_dataset(config.DATA_FILE_NAME)
        y = df[config.TARGET]
        y = EncodingTargetVariable(y)
        # Ensure the target variable is encoded correctly
        assert y[0] in [0, 1], "Target variable not encoded correctly"
        assert len(y.unique()) == 2, "Target variable not encoded correctly"

    except CustomException:
        pytest.fail("CustomException raised: Target encoding failed")
    except FileNotFoundError:
        pytest.fail("FileNotFoundError: Data file not found")
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")

#the pytest executes this fixture before below two tests, below test will use the output of this fixture for testing
@pytest.fixture
def single_prediction():
    try:
        test_data = load_dataset(config.TEST_FILE_NAME)
        X, y = separate_data(test_data)
        classification_pipeline = load_pipeline(config.MODEL_NAME)
        y_pred = classification_pipeline.predict(X)
        return y_pred[0]
    except FileNotFoundError:
        pytest.fail("FileNotFoundError: Data file not found")
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")
        
def test_generate_prediction_not_none(single_prediction):
    assert single_prediction is not None, "Prediction failed as it is None"

def test_check_dtype_of_prediction(single_prediction):
    assert single_prediction in [0, 1], "Prediction failed as it is not of type Integer"