from setuptools import setup, find_packages
from typing import List


def get_requirements(file_path:str)->List[str]:
    """"
    This function takes the path of requirements.txt as input and will load/install the libraries from their
    i.e. this function returns the list of requirements from the file_path mentioned.
    """
    requirements=[]
    with open(file_path,'r') as file_obj:
        requirements=file_obj.readlines()
    requirements = [x.replace("\n","").strip() for x in requirements]
    HYPHEN_E_DOT = '-e .'
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    
    return requirements
    

setup(
    name='EndToEndLoanPredictionProject',  
    version='0.0.1',
    author='Suhaib Mukhtar',
    author_email='suhaibmukhtar2@gmail.com',
    description='This project develops an end-to-end machine learning pipeline to predict loan approval likelihood using Python, MLflow for experiment tracking, and ML-pipelines for training and prediction. It incorporates Git/GitHub for version control, and Pytest for automated testing to ensure the validity of each step.',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires= get_requirements('requirements.txt')
)