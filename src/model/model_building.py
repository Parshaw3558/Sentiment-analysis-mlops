import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
from src.logger import logging
import mlflow
import mlflow.sklearn
import json
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/Parshaw3558/sentiment-analysis-mlops.mlflow")

dagshub.init(repo_owner="Parshaw3558",repo_name="sentiment-analysis-mlops",mlflow=True)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train the Logistic Regression model."""
    try:
        clf = LogisticRegression(C=1, solver='liblinear', penalty='l1')
        clf.fit(X_train, y_train)
        logging.info('Model training completed')
        return clf
    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
from src.logger import logging
import mlflow
import mlflow.sklearn
import json


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train the Logistic Regression model."""
    try:
        clf = LogisticRegression(C=1, solver='liblinear', penalty='l1')
        clf.fit(X_train, y_train)
        logging.info('Model training completed')
        return clf
    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:

        train_data = load_data('./data/processed/train_bow.csv')

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        mlflow.set_experiment("my-dvc-pipeline")

        print("Tracking URI:", mlflow.get_tracking_uri())

        with mlflow.start_run() as run:

            clf = train_model(X_train, y_train)

            save_model(clf, 'models/model.pkl')

            mlflow.sklearn.log_model(sk_model=clf,artifact_path="model")

            model_info = {"run_id": run.info.run_id,"model_path": "model"}

            with open("reports/experiment_info.json", "w") as f:
                json.dump(model_info, f, indent=4)

            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("solver", "liblinear")
            mlflow.log_param("penalty", "l1")
            mlflow.log_param("C", 1)

            logging.info("Model logged to MLflow successfully")

    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
