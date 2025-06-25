# Import libraries

import argparse
import glob
import os
import mlflow
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.datasets import load_diabetes

# define functions
def main(args):
    # TO DO: enable autologging
    mlflow.autolog()

    # read data
    df = pd.read_csv(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    model = train_model(args.reg_rate, X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(preds, y_test)
    metrics = {"mse": mse}
    print(metrics)

    model_name = "sklearn_regression_model.pkl"
    joblib.dump(value=model, filename=model_name)

def split_data(df):
    # split data into features and target variable
    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    return X_train, X_test, y_train, y_test 


def train_model(reg_rate, X_train, y_train):
    # train model
    model = LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)
    return model


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)

