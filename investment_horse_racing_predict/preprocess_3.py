import app_s3

df = app_s3.read_dataframe("preprocess_2.csv", index_col=0)
df.info()
df.head()

df_train = df.query("'2008-01-01'<=start_datetime<'2019-01-01'").dropna()
df_test = df.query("'2019-01-01'<=start_datetime<'2020-01-01'").dropna()

df_train_x = df_train.drop(["start_datetime", "result"], axis=1)
df_train_y = df_train["result"]

df_test_x = df_test.drop(["start_datetime", "result"], axis=1)
df_test_y = df_test["result"]

app_s3.write_dataframe(df_train_x, "preprocess_3.train_x.csv")
app_s3.write_dataframe(df_train_y, "preprocess_3.train_y.csv")
app_s3.write_dataframe(df_test_x, "preprocess_3.test_x.csv")
app_s3.write_dataframe(df_test_y, "preprocess_3.test_y.csv")
