import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import app_s3
from app_logging import logger


# Setting
INPUT_FILE_TRAIN_X = "preprocess_3_1.train_x.csv"
INPUT_FILE_TRAIN_Y = "preprocess_3_1.train_y.csv"
INPUT_FILE_TEST_X = "preprocess_3_1.test_x.csv"
INPUT_FILE_TEST_Y = "preprocess_3_1.test_y.csv"
MODEL_FILE = "model_2.preprocess_3_1.joblib"
OUTPUT_FILE_TRAIN_PREDICT = "predict.model_2.preprocess_3_1.train.csv"
OUTPUT_FILE_TEST_PREDICT = "predict.model_2.preprocess_3_1.test.csv"


# Load data
logger.info("Load data")

df_train_x = app_s3.read_dataframe(INPUT_FILE_TRAIN_X, index_col=0)
df_train_y = app_s3.read_dataframe(INPUT_FILE_TRAIN_Y, index_col=0)
df_test_x = app_s3.read_dataframe(INPUT_FILE_TEST_X, index_col=0)
df_test_y = app_s3.read_dataframe(INPUT_FILE_TEST_Y, index_col=0)


# Fit model
logger.info("Fit model")

model = RandomForestRegressor().fit(df_train_x.values, np.reshape(df_train_y.values, (-1)))

app_s3.write_sklearn_model(model, MODEL_FILE)
model = app_s3.read_sklearn_model(MODEL_FILE)


# Predict
logger.info("Predict")

df_train_y["predict"] = model.predict(df_train_x.values)
df_test_y["predict"] = model.predict(df_test_x.values)

app_s3.write_dataframe(df_train_y, OUTPUT_FILE_TRAIN_PREDICT)
app_s3.write_dataframe(df_test_y, OUTPUT_FILE_TEST_PREDICT)

train_score = {
    "mae": mean_absolute_error(df_train_y["result"].values, df_train_y["predict"].values),
    "mse": mean_squared_error(df_train_y["result"].values, df_train_y["predict"].values),
    "rmse": np.sqrt(mean_squared_error(df_train_y["result"].values, df_train_y["predict"].values)),
    "r2": r2_score(df_train_y["result"].values, df_train_y["predict"].values)
}

test_score = {
    "mae": mean_absolute_error(df_test_y["result"].values, df_test_y["predict"].values),
    "mse": mean_squared_error(df_test_y["result"].values, df_test_y["predict"].values),
    "rmse": np.sqrt(mean_squared_error(df_test_y["result"].values, df_test_y["predict"].values)),
    "r2": r2_score(df_test_y["result"].values, df_test_y["predict"].values)
}

logger.info("Score(train): %s" % train_score)
logger.info("Score(test): %s" % test_score)


logger.info("Finish")
