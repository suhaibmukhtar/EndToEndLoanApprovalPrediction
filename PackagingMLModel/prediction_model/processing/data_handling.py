import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

#sys.path:
# sys.path is a list in Python that contains the directories where Python looks for modules when you import them. By default, it includes standard library directories and any directories specified by the environment or by the user.
# If you want Python to be able to import modules from a specific directory that isn't included in the default sys.path, you can manually add it.

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from exception import CustomException
#the path above is added to the sys.path so that the modules in the parent directory can be imported, so that the config file can be imported easily without any errors

from prediction_model.config import config
#loading the dataset
def load_dataset(file_name: str) -> pd.DataFrame:
    try:
        file_path = os.path.join(PACKAGE_ROOT,config.DATASET_DIR, file_name)
        _data = pd.read_csv(file_path)
        #removing leading and trailing spaces from the column names
        _data.columns = [col.strip() for col in _data.columns]
        return _data
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
    
#saving the model to the disk
def save_pipeline(pipeline_to_save):
    try:
        model_path = os.path.join(PACKAGE_ROOT,config.SAVE_MODEL_PATH,config.MODEL_NAME)
        joblib.dump(pipeline_to_save, model_path)
    except Exception as e:
        raise CustomException(e,sys)
    
#loading the model from the disk
def load_pipeline():
    try:
        model_path = os.path.join(PACKAGE_ROOT,config.SAVE_MODEL_PATH,config.MODEL_NAME)
        loaded_model = joblib.load(model_path)
        return loaded_model
    except Exception as e:
        raise CustomException(e,sys)

