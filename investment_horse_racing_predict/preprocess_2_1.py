import app_s3

INPUT_FILE = "preprocess_1.csv"
OUTPUT_FILE = "preprocess_2_1.csv"


df = app_s3.read_dataframe(INPUT_FILE, index_col=0)
df.info()
df.head()

df = df.drop("race_id", axis=1)

df = df.drop("bracket_number", axis=1)

df = df.drop("horse_number", axis=1)

df = df.drop("horse_id", axis=1)

df = df.drop("arrival_time", axis=1)

df = df.drop("jockey_id", axis=1)

df = df.drop("trainer_id", axis=1)

df = df.drop("course_type", axis=1)

df = df.drop("weather", axis=1)

df = df.drop("course_condition", axis=1)

df = df.drop("gender", axis=1)

df = df.drop("birthday", axis=1)

df = df.drop("coat_color", axis=1)

df = df.drop("result_odds_win", axis=1)

df = df.drop("result_odds_place", axis=1)

df.info()
df.head()
app_s3.write_dataframe(df, OUTPUT_FILE)
