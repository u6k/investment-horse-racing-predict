from sklearn import ensemble

import app_s3

df_train_x = app_s3.read_dataframe("preprocess_3.train_x.csv", index_col=0)
df_train_y = app_s3.read_dataframe("preprocess_3.train_y.csv", index_col=0)
df_test_x = app_s3.read_dataframe("preprocess_3.test_x.csv", index_col=0)
df_test_y = app_s3.read_dataframe("preprocess_3.test_y.csv", index_col=0)

train_x = df_train_x.values
train_y = df_train_y.values
test_x = df_test_x.values
test_y = df_test_y.values

model = ensemble.RandomForestClassifier(n_estimators=500, criterion="entropy", max_depth=8, class_weight="balanced").fit(train_x, train_y)

df_test_y["predict"] = model.predict(test_x)

app_s3.write_dataframe(df_test_y, "predict_result.csv")
app_s3.write_sklearn_model(model, "model.joblib")
