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

## Push ML Application as a Package to Test PyPI

This guide explains how to push your ML application as a package to **Test PyPI**.

### Steps:

#### 1. Create an Account on Test PyPI
- Go to [test.pypi.org](https://test.pypi.org/) and click on **Register**.
- Provide your email, name, username, and password to create your account.

#### 2. Log in to Your Account
- After account creation, log in using your credentials.

#### 3. Generate an API Token
- To publish your application, you will need an API token.
- After logging in, go to the **Account Settings** page and generate the API token.

#### 4. Copy the API Token
- Once the token is generated, copy it as you will need it to authenticate during the upload process.

#### 5. Upload the Package Using Twine
- In your terminal, use the following command to upload your package:
  ```bash
  twine upload --repository-url https://test.pypi.org/legacy/ dist/your-package-name.tar.gz
  ```
  Replace `your-package-name.tar.gz` with the actual filename of your package.

#### 6. Authenticate with Your API Token
- After running the command, you will be prompted to enter your API token. Paste the token you copied earlier.

#### 7. Confirmation
- Once the upload is successful, you will receive a link that contains the location where your package is now hosted on Test PyPI.

By following these steps, you will be able to successfully push your ML application as a package to Test PyPI.