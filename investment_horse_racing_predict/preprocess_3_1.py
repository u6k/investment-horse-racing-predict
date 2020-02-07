import app_s3

INPUT_FILE = "preprocess_2_1.csv"
OUTPUT_FILE_TRAIN_X = "preprocess_3_1.train_x.csv"
OUTPUT_FILE_TRAIN_Y = "preprocess_3_1.train_y.csv"
OUTPUT_FILE_TEST_X = "preprocess_3_1.test_x.csv"
OUTPUT_FILE_TEST_Y = "preprocess_3_1.test_y.csv"


df = app_s3.read_dataframe(INPUT_FILE, index_col=0)
df.info()
df.head()

df_train = df.query("'2018-01-01'<=start_datetime<'2019-01-01'").dropna()
df_test = df.query("'2019-01-01'<=start_datetime<'2020-01-01'").dropna()

df_train_x = df_train.drop(["start_datetime", "result"], axis=1)
df_train_y = df_train["result"]

df_test_x = df_test.drop(["start_datetime", "result"], axis=1)
df_test_y = df_test["result"]

app_s3.write_dataframe(df_train_x, OUTPUT_FILE_TRAIN_X)
app_s3.write_dataframe(df_train_y, OUTPUT_FILE_TRAIN_Y)
app_s3.write_dataframe(df_test_x, OUTPUT_FILE_TEST_X)
app_s3.write_dataframe(df_test_y, OUTPUT_FILE_TEST_Y)
