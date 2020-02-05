import pandas as pd
import app_s3

INPUT_ORIGIN_FILE = "preprocess_1.csv"
INPUT_TRAIN_X_FILE = "preprocess_3_1.train_x.csv"
INPUT_TRAIN_Y_FILE = "preprocess_3_1.train_y.csv"
INPUT_TEST_X_FILE = "preprocess_3_1.test_x.csv"
INPUT_TEST_Y_FILE = "preprocess_3_1.test_y.csv"
INPUT_MODEL_FILE = "model_1.joblib"
OUTPUT_TRAIN_FILE = "simulate_trade_1.train.csv"
OUTPUT_TEST_FILE = "simulate_trade_1.test.csv"


def simulate_report(df_origin, df_predict):
    df = pd.concat([df_origin, df_predict["predict"]], axis=1)
    df = df.query("not predict.isnull()")

    df["win_buy_signal"] = ""
    df["win_buy_result"] = ""
    df["win_profit"] = 0.0
    df["place_buy_signal"] = ""
    df["place_buy_result"] = ""
    df["place_profit"] = 0.0

    # TODO
    df["result_odds_win"].fillna(1, inplace=True)
    df["result_odds_place"].fillna(1, inplace=True)

    for id in df.index:
        if df.at[id, "predict"] == 1:
            df.at[id, "win_buy_signal"] = "buy"

            if df.at[id, "result"] == 1:
                df.at[id, "win_buy_result"] = "win"
                df.at[id, "win_profit"] = df.at[id, "result_odds_win"] - 1.0
            else:
                df.at[id, "win_buy_result"] = "lose"
                df.at[id, "win_profit"] = -1.0

        if df.at[id, "predict"] <= 3:
            df.at[id, "place_buy_signal"] = "buy"

            if df.at[id, "result"] <= 3:
                df.at[id, "place_buy_result"] = "win"
                df.at[id, "place_profit"] = df.at[id, "result_odds_place"] - 1.0
            else:
                df.at[id, "place_buy_result"] = "lose"
                df.at[id, "place_profit"] = -1.0

    app_s3.write_dataframe(df, "simulate_result.csv")

    simulate_result = {
        "win_buy_count": len(df.query("win_buy_signal == 'buy'")),
        "win_win_count": len(df.query("win_buy_result == 'win'")),
        "win_lose_count": len(df.query("win_buy_result == 'lose'")),
        "win_profit_total": df.query("win_buy_result == 'win'")["win_profit"].sum(),
        "win_loss_total": df.query("win_buy_result == 'lose'")["win_profit"].sum(),
        "win_profit_average": df.query("win_buy_result == 'win'")["win_profit"].mean(),
        "win_loss_average": df.query("win_buy_result == 'lose'")["win_profit"].mean(),
        "place_buy_count": len(df.query("place_buy_signal == 'buy'")),
        "place_win_count": len(df.query("place_buy_result == 'win'")),
        "place_lose_count": len(df.query("place_buy_result == 'lose'")),
        "place_profit_total": df.query("place_buy_result == 'win'")["place_profit"].sum(),
        "place_loss_total": df.query("place_buy_result == 'lose'")["place_profit"].sum(),
        "place_profit_average": df.query("win_buy_result == 'win'")["place_profit"].mean(),
        "place_loss_average": df.query("win_buy_result == 'lose'")["place_profit"].mean(),
    }
    simulate_result["win_win_rate"] = simulate_result["win_win_count"] / simulate_result["win_buy_count"]
    simulate_result["win_profit_factor"] = simulate_result["win_profit_total"] / abs(simulate_result["win_loss_total"])
    simulate_result["win_payoff_ratio"] = simulate_result["win_profit_average"] / abs(simulate_result["win_loss_average"])
    simulate_result["place_win_rate"] = simulate_result["place_win_count"] / simulate_result["place_buy_count"]
    simulate_result["place_profit_factor"] = simulate_result["place_profit_total"] / abs(simulate_result["place_loss_total"])
    simulate_result["place_payoff_ratio"] = simulate_result["place_profit_average"] / abs(simulate_result["place_loss_average"])

    print(simulate_result)


df_origin = app_s3.read_dataframe(INPUT_ORIGIN_FILE, index_col=0)
df_train_x = app_s3.read_dataframe(INPUT_TRAIN_X_FILE, index_col=0)
df_train_y = app_s3.read_dataframe(INPUT_TRAIN_Y_FILE, index_col=0)
df_test_x = app_s3.read_dataframe(INPUT_TEST_X_FILE, index_col=0)
df_test_y = app_s3.read_dataframe(INPUT_TEST_Y_FILE, index_col=0)

model = app_s3.read_sklearn_model(INPUT_MODEL_FILE)

df_train_y["predict"] = model.predict(df_train_x.values)
df_test_y["predict"] = model.predict(df_test_x.values)

app_s3.write_dataframe(df_train_y, OUTPUT_TRAIN_FILE)
app_s3.write_dataframe(df_test_y, OUTPUT_TEST_FILE)

simulate_report(df_origin, df_train_y)
simulate_report(df_origin, df_test_y)
