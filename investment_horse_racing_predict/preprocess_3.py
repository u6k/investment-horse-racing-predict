import app_s3

df = app_s3.read_dataframe("preprocess_2.csv", index_col=0)
df.info()
df.head()

df_train = df.query("'2008-01-01'<=start_datetime<'2019-01-01'")
df_test = df.query("'2019-01-01'<=start_datetime<'2020-01-01'")

df_x_train = df_train.drop(["start_datetime", "result"], axis=1)
df_y_train = df_train["result"]

df_x_test = df_test.drop(["start_datetime", "result"], axis=1)
df_y_test = df_test["result"]

app_s3.write_dataframe(df_x_train, "preprocess_3.x_train.csv")
app_s3.write_dataframe(df_y_train, "preprocess_3.y_train.csv")
app_s3.write_dataframe(df_x_test, "preprocess_3.x_test.csv")
app_s3.write_dataframe(df_y_test, "preprocess_3.y_test.csv")
