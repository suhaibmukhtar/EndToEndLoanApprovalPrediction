import os
from prediction_model.config import config

# Read the version from the VERSION file
version_file_path = os.path.join(os.path.dirname(__file__), 'VERSION')
with open(version_file_path) as version_file:
    __version__ = version_file.read().strip()