import argparse

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb
import matplotlib as mpl

import os
import mlflow
import mlflow.lightgbm
import pandas as pd

mpl.use("Agg")

import logging

# dotenv.load_dotenv()
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
os.environ['MLFLOW_S3_ENDPOINT_URL']='http://localhost:9000'

os.environ['AWS_ACCESS_KEY_ID']='minio'
os.environ['AWS_SECRET_ACCESS_KEY']='minio123'



def parse_args():
    parser = argparse.ArgumentParser(description="LightGBM example")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="learning rate to update step size at each boosting step (default: 0.3)",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=1.0,
        help="subsample ratio of columns when constructing each tree (default: 1.0)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="subsample ratio of the training instances (default: 1.0)",
    )
    return parser.parse_args()


def main():
    # parse command-line arguments
    args = parse_args()

    # prepare train and test data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    #new_data = 'D:/Cu_Ni_Ag_Database.xlsx'
    #data = pd.read_excel(new_data)
    #X = data.iloc[:, [0,1,2]]
    #y = data[data.columns[3:4]]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # enable auto logging
    mlflow.lightgbm.autolog()

    train_set = lgb.Dataset(X_train, label=y_train)

    with mlflow.start_run():

        # train model
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "learning_rate": args.learning_rate,
            "metric": "multi_logloss",
            "colsample_bytree": args.colsample_bytree,
            "subsample": args.subsample,
            "seed": 42,
        }
        model = lgb.train(
            params, train_set, num_boost_round=10, valid_sets=[train_set], valid_names=["train"]
        )

        

        # evaluate model
        y_proba = model.predict(X_test)
        y_pred = y_proba.argmax(axis=1)
        loss = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        print("accurary score", acc)

        # log metrics
        mlflow.log_metrics({"log_loss": loss, "accuracy": acc})


if __name__ == "__main__":
    main()