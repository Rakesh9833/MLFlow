
# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn import datasets
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 


import logging

# dotenv.load_dotenv()
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
#os.environ['MLFLOW_TRACKING_URI']='http://172.22.56.81:5000/'
os.environ['MLFLOW_S3_ENDPOINT_URL']='http://localhost:9000'

os.environ['AWS_ACCESS_KEY_ID']='minio'
os.environ['AWS_SECRET_ACCESS_KEY']='minio123'

print("here")

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average='weighted')
    recall = recall_score(actual, pred, average='macro')
    f1 = f1_score(actual, pred, average='macro')

    return rmse, mae, r2, accuracy, precision, recall, f1


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    new_data = 'C:/Users/rakes/Downloads/archive/IRIS.csv'
    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )
    try:
        #data = pd.read_csv(csv_url, sep=";")
        data = pd.read_csv(new_data)
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )


    wine = datasets.load_wine()

    print(wine.DESCR)

    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['label'] = wine.target
    
    print("dataset details", df)  

    X, y = load_wine(return_X_y=True, as_frame=True)

 
    reg_lambda=1
    gamma=0

    model = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder =False, reg_lambda = reg_lambda, gamma = gamma)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    with mlflow.start_run():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        (rmse, mae, r2, accuracy, precision, recall, f1) = eval_metrics(y_test, pred)


        mlflow.log_param("reg_lambda", reg_lambda)
        mlflow.log_param("gamma", gamma)
        #mlflow.log_metric("rmse", rmse)
        #mlflow.log_metric("r2", r2)
        #mlflow.log_metric("mae", mae)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        #print("  RMSE: %s" % rmse)
        #print("  MAE: %s" % mae)
        #print("  R2: %s" % r2)
        print("  accuracy: %s" % accuracy)
        print("  precision: %s" % precision)
        print("  recall: %s" % recall)
        print("  f1: %s" % f1)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print('---------------------------------------------')
        if tracking_url_type_store != "file":
            print('Here-------------------------------------')
            mlflow.sklearn.log_model(model, "model", registered_model_name="Wine prediction using Xgboost Classifier")
            print('Logging---------------------------------------------')
        else:
            mlflow.sklearn.log_model(model, "model")

            

