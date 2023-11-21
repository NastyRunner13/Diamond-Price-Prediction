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

            model_report: dict = evaluate_model(
                X_train, y_train, X_test, y_test, models
            )

            print("\n==============================================")
            logging.info(f"Model Report : {model_report}")

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(
                f"Best Model Found, Model Name : {best_model_name}, Accuracy Score : {best_model_score}"
            )
            print("\n==============================================")
            logging.info(
                f"Best Model Found, Model Name : {best_model_name}, Accuracy Score : {best_model_score}"
            )

            save_object(
                file_path=self.model_trainer_config.trainer_model_file_path,
                obj=best_model,
            )

        except Exception as e:
            raise CustomException(e, sys)
