from sklearn import ensemble, model_selection

import app_s3

INPUT_FILE_TRAIN_X = "preprocess_3_1.train_x.csv"
INPUT_FILE_TRAIN_Y = "preprocess_3_1.train_y.csv"
INPUT_FILE_TEST_X = "preprocess_3_1.test_x.csv"
INPUT_FILE_TEST_Y = "preprocess_3_1.test_y.csv"
MODEL_FILE = "model_1.joblib"


df_train_x = app_s3.read_dataframe(INPUT_FILE_TRAIN_X, index_col=0)
df_train_y = app_s3.read_dataframe(INPUT_FILE_TRAIN_Y, index_col=0)
df_test_x = app_s3.read_dataframe(INPUT_FILE_TEST_X, index_col=0)
df_test_y = app_s3.read_dataframe(INPUT_FILE_TEST_Y, index_col=0)

train_x = df_train_x.values
train_y = df_train_y.values
test_x = df_test_x.values
test_y = df_test_y.values

# best_params: {'criterion': 'gini', 'max_depth': 8, 'n_estimators': 1000}
# parameters = {
#     "n_estimators": [100, 200, 500, 750, 1000],
#     "criterion": ["gini", "entropy"],
#     "max_depth": [4, 8, 16, 32, 64],
# }
#
# clf = model_selection.GridSearchCV(
#     ensemble.RandomForestClassifier(),
#     parameters,
#     cv=5,
#     n_jobs=-1,
#     verbose=1
# )
#
# print(f"best_params: {clf.best_params_}")
#
# model = clf.best_estimator_

parameters = {
    "n_estimators": [128, 256, 512, 1024, 2048, 4096, 8192],
    "max_depth": [4, 8, 16, 64],
}

clf = model_selection.GridSearchCV(
    ensemble.RandomForestRegressor(),
    parameters,
    cv=5,
    n_jobs=-1,
    verbose=1
)

clf.fit(train_x, train_y)

print(f"best_params: {clf.best_params_}")

model = clf.best_estimator_

# model = ensemble.RandomForestClassifier(criterion="gini", max_depth=8, n_estimators=1000)
# model.fit(train_x, train_y)

app_s3.write_sklearn_model(model, MODEL_FILE)
