
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
#from xgboost import XGBRegressor


import logging

# dotenv.load_dotenv()
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
os.environ['MLFLOW_S3_ENDPOINT_URL']='http://localhost:9000'

os.environ['AWS_ACCESS_KEY_ID']='minio'
os.environ['AWS_SECRET_ACCESS_KEY']='minio123'

print("here")

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    new_data = 'D:/Cu_Ni_Ag_Database.xlsx'
    #df = pd.read_excel(new_data)
    #print("this is the result", df)
    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )
    try:
        #data = pd.read_csv(csv_url, sep=";")
        data = pd.read_excel(new_data)
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    reg_lambda=1
    gamma=0

    #model = xgb.XGBRegressor()
    model = xgb.XGBRegressor(n_estimators=20, reg_lambda=1, gamma=0, max_depth=3)

    #model = XGBRegressor()

    X = data.iloc[:, [0,1,2]]
    y = data[data.columns[3:4]]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    #history = model.fit(X_train, y_train)


    with mlflow.start_run():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        (rmse, mae, r2) = eval_metrics(y_test, pred)


        #mlflow.log_param("alpha", alpha)
        #mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("reg_lambda", reg_lambda)
        mlflow.log_param("gamma", gamma)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)


        #x_ax = range(len(y_test))
        #plt.plot(x_ax, y_test, label="original")
        #plt.plot(x_ax, y_pred, label="predicted")
        #plt.title("MSE test and predicted data")
        #plt.legend()
        #plt.show()



        #lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        #lr.fit(train_x, train_y)

        #predicted_qualities = lr.predict(test_x)

        #(rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        #print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        #tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        #tracking_url_type_store = urlparse(mlflow.set_tracking_uri("http://localhost:5000")).scheme
        #mlflow.set_tracking_uri("http://localhost:5000") 
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        #mlflow.set_tracking_uri("http://localhost:5000")
        #print("---", mlflow.get_tracking_uri())
        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model, "model", registered_model_name="Material_Property_Prediction_Xgboost")
        else:
            mlflow.sklearn.log_model(model, "model")

            

