import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import evaluate_model
from src.utils import save_object

import os
import sys


@dataclass
class ModelTrainerConfig:
    trainer_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting Dependent and Independent variable")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet(),
                "DecisonTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "KNeighbors": KNeighborsRegressor(),
            }

            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)

            print("\n==============================================")
            logging.info(f"Model Report : {model_report}")

            # Initialize variables to store the model with the highest F1-score and its value
            max_accuracy_model = None
            max_accuracy_score = 0

            # Iterate through the models in the report dictionary
            for model, metrics in model_report.items():
                accuracy = metrics["Accuracy"]

                # Update if the current model has a higher F1-score
                if accuracy > max_accuracy_score:
                    max_accuracy_score = accuracy
                    max_accuracy_model = model

            print(
                f"Best Model Found, Model Name : {max_accuracy_model}, Accuracy Score : {max_accuracy_score}"
            )
            print("\n==============================================")
            logging.info(
                f"Best Model Found, Model Name : {max_accuracy_model}, Accuracy Score : {max_accuracy_score}"
            )

            save_object(
                file_path=self.model_trainer_config.trainer_model_file_path,
                obj=max_accuracy_model,
            )

        except Exception as e:
            raise CustomException(e, sys)
