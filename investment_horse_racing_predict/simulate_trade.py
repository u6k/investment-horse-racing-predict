import pandas as pd
import app_s3

df_origin = app_s3.read_dataframe("preprocess_1.csv", index_col=0)
df_test_x = app_s3.read_dataframe("preprocess_3.test_x.csv", index_col=0)
df_test_y = app_s3.read_dataframe("preprocess_3.test_y.csv", index_col=0)

clf = app_s3.read_sklearn_model("model.joblib")

df_test_y["predict"] = clf.predict(df_test_x.values)

app_s3.write_dataframe(df_test_y, "predict_result.csv")

df = pd.concat([df_origin, df_test_y["predict"]], axis=1)
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
