from setuptools import setup, find_packages
from typing import List
import os

def get_requirements(file_path: str) -> List[str]:
    """
    This function takes the path of requirements.txt as input and returns a list of libraries from it.
    It removes any unwanted lines (e.g., -e . for editable installs).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' was not found!")

    requirements = []
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()

    requirements = [x.strip() for x in requirements if x.strip()]  # Remove empty lines and strip spaces

    HYPHEN_E_DOT = '-e .'
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name='EndToEndLoanPredictionProject',  
    version='0.0.1',
    author='Suhaib Mukhtar',
    author_email='suhaibmukhtar2@gmail.com',
    description=(
        'This project develops an end-to-end machine learning pipeline to predict loan approval '
        'likelihood using Python, MLflow for experiment tracking, and ML-pipelines for training and '
        'prediction. It incorporates Git/GitHub for version control, and Pytest for automated testing '
        'to ensure the validity of each step.'
    ),
    packages=find_packages(),
    install_requires=get_requirements('./requirements.txt'),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ]
)
