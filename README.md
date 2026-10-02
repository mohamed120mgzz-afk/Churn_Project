# Bank Customer Churn Prediction

A Machine Learning project that predicts whether a bank customer is likely to leave the bank based on customer demographics, account information, and banking activity.

The project includes tuned classification models, a preprocessing pipeline, structured input validation using Pydantic, and an inference module designed to support model predictions.

## Table of Contents

* [Project Overview](#project-overview)
* [Objectives](#objectives)
* [Machine Learning Models](#machine-learning-models)
* [Project Structure](#project-structure)
* [Dataset](#dataset)
* [Features](#features)
* [Technologies Used](#technologies-used)
* [Installation](#installation)
* [Usage](#usage)
* [Input Example](#input-example)
* [Model Artifacts](#model-artifacts)
* [Future Improvements](#future-improvements)
* [Author](#author)

## Project Overview

Customer churn occurs when customers stop using a company's products or services. In the banking industry, identifying customers who may leave can help businesses better understand customer behavior and develop customer retention strategies.

This project uses supervised Machine Learning classification techniques to predict customer churn using the Bank Customer Churn dataset.

The repository contains trained classification models, a preprocessing artifact, and Python modules for validating customer data and handling inference.

## Objectives

* Predict whether a bank customer is likely to churn.
* Apply data preprocessing techniques before model inference.
* Train and tune classification models.
* Handle class imbalance using class weights or sample weights where appropriate.
* Validate incoming customer data using Pydantic.
* Organize model artifacts and inference logic into reusable modules.
* Prepare the project for integration with an API or other prediction interface.

## Machine Learning Models

The repository contains the following trained model artifacts:

| Model                          | Artifact              |
| ------------------------------ | --------------------- |
| Tuned Random Forest Classifier | `forest_tuned.pkl`    |
| Tuned XGBoost Classifier       | `xgb_tuned.pkl`       |
| Data Preprocessing Pipeline    | `preprocessor.joblib` |

Random Forest and XGBoost are tree-based classification algorithms that can learn relationships between customer characteristics and churn outcomes.

Model performance can be evaluated using the following classification metrics:

* **Precision:** Measures the proportion of predicted churners who actually churned.
* **Recall:** Measures the proportion of actual churners correctly identified.
* **F1-score:** The harmonic mean of precision and recall.
* **Confusion Matrix:** Shows the distribution of correct and incorrect predictions.

Because customer churn datasets may have imbalanced target classes, accuracy alone may not adequately describe model performance.

The final model selection should be based on measured evaluation results rather than assumptions about which algorithm performs better.

## Project Structure

```text
ml_project/
│
├── .vscode/
│   └── settings.json
│
├── Dataset/
│   └── Churn_Modelling.csv
│
├── models/
│   ├── forest_tuned.pkl
│   ├── preprocessor.joblib
│   └── xgb_tuned.pkl
│
├── NoteBook/
│   └── notebook.ipynb
│
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── CustomerData.py
│   └── inferance.py
│
├── main.py
├── requirements.txt
└── README.md
```

### Directory Description

* **`Dataset/`** — Contains the dataset used for data analysis and model development.
* **`models/`** — Stores the trained classification models and preprocessing pipeline.
* **`NoteBook/`** — Contains the Jupyter Notebook used for experimentation and model development.
* **`utils/config.py`** — Intended for shared configuration and file paths.
* **`utils/CustomerData.py`** — Defines the customer input schema and validation rules using Pydantic.
* **`utils/inferance.py`** — Intended to contain model loading, preprocessing, and prediction logic.
* **`main.py`** — Project entry point, which can host the API application.
* **`requirements.txt`** — Lists the Python dependencies required to run the project.

## Dataset

The project uses `Churn_Modelling.csv`, a bank customer dataset commonly used for customer churn classification.

The dataset contains customer demographic information, account characteristics, and a target variable indicating whether a customer exited the bank.

### Input Features

| Feature           | Description                                         |
| ----------------- | --------------------------------------------------- |
| `CreditScore`     | Customer's credit score                             |
| `Geography`       | Customer's country                                  |
| `Gender`          | Customer's gender                                   |
| `Age`             | Customer's age                                      |
| `Tenure`          | Number of years the customer has been with the bank |
| `Balance`         | Customer's account balance                          |
| `NumOfProducts`   | Number of bank products held                        |
| `HasCrCard`       | Whether the customer has a credit card              |
| `IsActiveMember`  | Whether the customer is an active member            |
| `EstimatedSalary` | Customer's estimated annual salary                  |

### Target Variable

**`Exited`**

* `0` — The customer did not leave the bank.
* `1` — The customer left the bank.

The target variable is used during supervised model training and must not be included among the input features supplied to the prediction model.

The feature set and preprocessing steps used during inference must match those used during training.

## Technologies Used

* **Python** — Core programming language.
* **Pandas** — Data manipulation and analysis.
* **NumPy** — Numerical operations.
* **Scikit-learn** — Preprocessing, model evaluation, and hyperparameter tuning.
* **XGBoost** — Gradient-boosted tree classification.
* **Pydantic** — Input data validation and schema enforcement.
* **Joblib / Pickle** — Serialization and loading of trained model artifacts.
* **Jupyter Notebook** — Data exploration and experimentation.
* **FastAPI** — API development, if implemented in the application.
* **Uvicorn** — ASGI server for running the FastAPI application.

## Installation

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd ml_project
```

Replace `<YOUR_REPOSITORY_URL>` with your GitHub repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure that the installed library versions are compatible with the versions used to train and save the model artifacts.

## Usage

The project is organized around saved models, a preprocessing pipeline, and utility modules for input validation and inference.

The general prediction workflow is:

1. Receive customer information.
2. Validate the input using the Pydantic customer schema.
3. Load the trained model and preprocessing pipeline.
4. Transform the input using the saved preprocessing pipeline.
5. Pass the transformed features to the selected classifier.
6. Return the predicted churn class.

The prediction can be interpreted as follows:

* `0` — Predicted not to churn.
* `1` — Predicted to churn.

These labels represent the model's predictions, not guaranteed future customer behavior.

The exact execution command depends on the implementation of `main.py` and `utils/inferance.py`.

## Input Example

The following JSON object represents an example customer record:

```json
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Male",
  "Age": 30,
  "Tenure": 5,
  "Balance": 75000.5,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 85000.75
}
```

### Input Validation Rules

The customer schema defines the following constraints:

* `Geography` must be `France`, `Spain`, or `Germany`.
* `Gender` must be `Male` or `Female`.
* `Age` must be between 18 and 100.
* `Tenure` must be between 0 and 10.
* `Balance` must be non-negative.
* `NumOfProducts` must be between 1 and 4.
* `HasCrCard` must be `0` or `1`.
* `IsActiveMember` must be `0` or `1`.
* `CreditScore` must be an integer.
* `EstimatedSalary` must be non-negative.

These validation rules help ensure that incoming data follows the expected input schema.

## Model Artifacts

The `models/` directory contains the serialized artifacts used for inference.

| File                  | Purpose                           |
| --------------------- | --------------------------------- |
| `forest_tuned.pkl`    | Saved Random Forest classifier    |
| `xgb_tuned.pkl`       | Saved XGBoost classifier          |
| `preprocessor.joblib` | Saved data preprocessing pipeline |

For consistent predictions, use the same preprocessing logic and feature order as during training. Do not fit a new preprocessing pipeline on individual inference requests.

**Security note:** Only load Pickle or Joblib artifacts from trusted sources. Deserializing untrusted files can execute arbitrary code.

## Future Improvements

Potential future improvements include:

* Develop and document a FastAPI prediction endpoint.
* Add automated tests for input validation and inference.
* Compare Random Forest and XGBoost using a held-out test set.
* Document the selected model and its measured evaluation metrics.
* Add prediction probabilities and a configurable decision threshold.
* Improve error handling and logging.
* Add a reproducible training pipeline and model versioning.
* Pin dependency versions for reproducibility.
* Add Docker support for deployment.
* Set up continuous integration for automated testing.
* Add a project license.

## Author

**Mohamed Ahmed Saad Shabeb**

Computer Science Student | AI & Machine Learning

GitHub: <https://github.com/mohamed120mgzz-afk>

LinkedIn: <linkedin.com/in/mohamed-ahmed-22448735a>

---

* This project is intended for educational and demonstration purposes. Predictions should not be treated as definitive assessments of individual customers or used as the sole basis for consequential banking decisions.*
