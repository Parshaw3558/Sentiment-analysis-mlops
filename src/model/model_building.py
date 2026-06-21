import os
import json
import pickle

import dagshub
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from src.logger import logging


# ---------------------------
# DagsHub + MLflow Setup
# ---------------------------

token = os.getenv("DAGSHUB_TOKEN")

if token:
    try:
        dagshub.auth.add_app_token(token)

        dagshub.init(
            repo_owner="Parshaw3558",
            repo_name="sentiment-analysis-mlops",
            mlflow=True
        )

        mlflow.set_tracking_uri(
            "https://dagshub.com/Parshaw3558/sentiment-analysis-mlops.mlflow"
        )

        print("DagsHub MLflow connected successfully")

    except Exception as e:
        print(f"DagsHub initialization failed: {e}")
else:
    print("DAGSHUB_TOKEN not found. Running without DagsHub tracking.")


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded from {file_path}")
        return df

    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> LogisticRegression:

    try:
        clf = LogisticRegression(
            C=1,
            solver="liblinear",
            penalty="l1"
        )

        clf.fit(X_train, y_train)

        logging.info("Model training completed")

        return clf

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise


def save_model(model, file_path: str):

    try:
        with open(file_path, "wb") as f:
            pickle.dump(model, f)

        logging.info(f"Model saved to {file_path}")

    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise


def main():

    try:

        train_data = load_data(
            "./data/processed/train_bow.csv"
        )

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        mlflow.set_experiment("my-dvc-pipeline")

        print("Tracking URI:", mlflow.get_tracking_uri())

        with mlflow.start_run() as run:

            clf = train_model(
                X_train,
                y_train
            )

            save_model(
                clf,
                "models/model.pkl"
            )

            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model"
            )

            model_info = {
                "run_id": run.info.run_id,
                "model_path": "model"
            }

            with open(
                "reports/experiment_info.json",
                "w"
            ) as f:
                json.dump(
                    model_info,
                    f,
                    indent=4
                )

            mlflow.log_param(
                "model_type",
                "LogisticRegression"
            )

            mlflow.log_param(
                "solver",
                "liblinear"
            )

            mlflow.log_param(
                "penalty",
                "l1"
            )

            mlflow.log_param(
                "C",
                1
            )

            logging.info(
                "Model logged successfully"
            )

    except Exception as e:

        logging.error(
            f"Failed to complete model building: {e}"
        )

        raise


if __name__ == "__main__":
    main()