import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

#sys.path:
# sys.path is a list in Python that contains the directories where Python looks for modules when you import them. By default, it includes standard library directories and any directories specified by the environment or by the user.
# If you want Python to be able to import modules from a specific directory that isn't included in the default sys.path, you can manually add it.

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from logger import logging
from exception import CustomException
logging.info(f"Obtained PACKAGE_ROOT Successfully")
#the path above is added to the sys.path so that the modules in the parent directory can be imported, so that the config file can be imported easily without any errors

from prediction_model.config import config
#loading the dataset
def load_dataset(file_name: str) -> pd.DataFrame:
    try:
        file_path = os.path.join(PACKAGE_ROOT,config.DATASET_DIR, file_name)
        _data = pd.read_csv(file_path)
        #removing leading and trailing spaces from the column names
        _data.columns = [col.strip() for col in _data.columns]
        #creating a new column 'total_assets_value' by adding  all asset value columns
        _data['total_assets_value'] = _data['residential_assets_value'] + _data['commercial_assets_value'] + _data['luxury_assets_value'] + _data['bank_asset_value']
        logging.info("New column 'total_assets_value' created successfully")
        return _data[config.FEATURES]
    except Exception as e:
        raise CustomException(e,sys)
#Creating matrix of features and dependent variable vector i.e. separating the features and target variable
def separate_data(df: pd.DataFrame) -> tuple:
    try:
        X = df.drop(config.TARGET, axis=1)
        y = df[config.TARGET]
        return X, y
    except Exception as e:
        raise CustomException(e,sys) 
    
#splitting the dataset into training and testing set
def data_split_strategy(X: pd.DataFrame, y: pd.Series) -> tuple:
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE,stratify=y, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise CustomException(e,sys)

if __name__ == '__main__':
    logging.info("Starting Data Handling")
    #loading the dataset
    df = load_dataset(config.DATA_FILE_NAME)
    logging.info("Dataset loaded successfully")
    logging.info(f"Dataset shape:{df.shape}")
    logging.info(f"Columns in the dataset:{df.columns}")
    logging.info(f"Columns with missing values:{df.isna().sum()}")
    logging.info(f"duplicate rows in the dataset:{df.duplicated().sum()}")
    X, y = separate_data(df)
    logging.info("Data separated successfully")
    X_train, X_test, y_train, y_test = data_split_strategy(X, y)
    logging.info("Data split into training and testing set successfully")
    logging.info(f"X-training set shape:{X_train.shape}")
    logging.info(f"X-testing set shape:{X_test.shape}")
    logging.info(f"y-training set shape:{y_train.value_counts()}")