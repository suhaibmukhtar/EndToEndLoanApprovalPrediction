import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from logger import logging
from exception import CustomException
from sklearn.preprocessing import OneHotEncoder

def CreateCustomColumns(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df[config.CUSTOM_COLUMN_NAME] = df[config.COLUMNS_TO_MERGE[0]] + df[config.COLUMNS_TO_MERGE[1]] + df[config.COLUMNS_TO_MERGE[2]] + df[config.COLUMNS_TO_MERGE[3]]
        logging.info(f"New column {config.CUSTOM_COLUMN_NAME} created successfully")
        return df
    except Exception as e:
        raise CustomException(e,sys)
    
def ColumnsToDrop(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(config.COLUMNS_TO_DROP, axis=1,inplace=True)
        return df
    except Exception as e:
        raise CustomException(e,sys)
    
def EncodingTarget(text: str) -> int:
    try:
        if text == ' Approved':
            return 1
        else:
            return 0
    except Exception as e:
        raise CustomException(e,sys)
    
def EncodingTargetVariable(y: pd.Series) -> pd.Series:
    try:
        # Map the categorical values to numeric values
        y_encoded = y.map(EncodingTarget)
        return y_encoded
    except Exception as e:
        raise CustomException(e, sys)
    
def TransformingNumericFeatures(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df[config.LOG_TRANSFORMATION] = np.log1p(df[config.LOG_TRANSFORMATION])
        return df
    except Exception as e:
        raise CustomException(e,sys)