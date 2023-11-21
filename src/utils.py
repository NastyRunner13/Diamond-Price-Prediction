import os
import sys
import pickle
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            # TRAIN MODEL
            model.fit(X_train, y_train)

            # PREDICT TRAINING AND TESTING DATA
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # GET R2SCORE FOR TRAIN AND TEST DATA
            # train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            test_model_mae = mean_absolute_error(y_test, y_test_pred)
            test_model_mse = mean_squared_error(y_test, y_test_pred)

            report[list(models.keys())[i]] = {
                "Accuracy": test_model_score,
                "Mean Absolute Error": test_model_mae,
                "Mean Squared Error": test_model_mse,
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Exception Occured in load_obj function utils")
        raise CustomException(e, sys)
