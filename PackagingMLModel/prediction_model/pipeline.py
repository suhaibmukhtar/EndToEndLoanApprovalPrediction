import os
import sys
from sklearn.pipeline import Pipeline
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import prediction_model.processing.data_preprocessing as dp
from sklearn.linear_model import LogisticRegression


#pipeline 
categorical_features_to_encode = config.CATEGORICAL_FEATURES_TO_ENCODE
numeric_features_to_transform = config.LOG_TRANSFORMATION
Target_var = config.TARGET

# Features that do not need transformation or encoding
features_without_transformation = [feature for feature in config.FEATURES if feature not in categorical_features_to_encode and feature not in numeric_features_to_transform and feature not in Target_var]

categorical_pipeline = Pipeline([
    ('Encoding_categorical_features', OneHotEncoder(handle_unknown='ignore',sparse_output=False,drop='first')),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', categorical_pipeline, categorical_features_to_encode),
        ('features_without_transformation', 'passthrough', features_without_transformation)
    ]
)   

classification_pipeline = Pipeline([
    ('preprocessor', preprocessor),  # Preprocessing step
    ('model', LogisticRegression(random_state=42))  # Logistic regression model
])
