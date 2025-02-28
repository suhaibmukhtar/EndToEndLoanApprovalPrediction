# EndToEndLoadApprovalPrediction
This project develops an end-to-end machine learning pipeline to predict loan approval likelihood using Python, MLflow for experiment tracking, and ML-pipelines for training and prediction. It incorporates Git/GitHub for version control, and Pytest for automated testing to ensure the validity of each step.

## Creating and Distributing Your Python Package

After creating your `setup.py` file, you can generate a distributable package by running the following command:

```bash
python setup.py sdist
```

This will create a `dist` directory in your project folder, which contains a `.tar.gz` file. This file is the source distribution of your package and can be used as an installable package.

### Installing the Package Locally

To install the package locally, run the following command:

```bash
pip install path/to/package.tar.gz
```

Make sure to replace `path/to/package.tar.gz` with the actual path to the `.tar.gz` file.

### Sharing the Package

You can also share the `.tar.gz` file with others. They can install the package on their system using the following command:

```bash
pip install path/to/package.tar.gz
```

This allows easy distribution of your package without needing to upload it to a package index like PyPI.

Alternatively, the package can be uploaded to PyPI, allowing others to directly install it from there.