from sklearn import ensemble, model_selection

import app_s3

df_train_x = app_s3.read_dataframe("preprocess_3.train_x.csv", index_col=0)
df_train_y = app_s3.read_dataframe("preprocess_3.train_y.csv", index_col=0)
df_test_x = app_s3.read_dataframe("preprocess_3.test_x.csv", index_col=0)
df_test_y = app_s3.read_dataframe("preprocess_3.test_y.csv", index_col=0)

train_x = df_train_x.values
train_y = df_train_y.values
test_x = df_test_x.values
test_y = df_test_y.values

# best_params: {'criterion': 'gini', 'max_depth': 8, 'n_estimators': 1000}
parameters = {
    "n_estimators": [100, 200, 500, 750, 1000],
    "criterion": ["gini", "entropy"],
    "max_depth": [4, 8, 16, 32, 64],
}

clf = model_selection.GridSearchCV(
    ensemble.RandomForestClassifier(),
    parameters,
    cv=5,
    n_jobs=-1,
    verbose=1
)

clf.fit(train_x, train_y)

print(f"best_params: {clf.best_params_}")

model = clf.best_estimator_

app_s3.write_sklearn_model(model, "model.joblib")
