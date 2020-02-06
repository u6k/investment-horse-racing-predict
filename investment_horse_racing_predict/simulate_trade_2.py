import pandas as pd

import app_s3
from app_logging import logger


# Setting
INPUT_FILE_ORIGIN = "preprocess_1.csv"
INPUT_FILE_TRAIN_PREDICT = "predict.model_1.preprocess_3_1.train.csv"
INPUT_FILE_TEST_PREDICT = "predict.model_1.preprocess_3_1.test.csv"
OUTPUT_FILE_TRAIN_SIMULATE = "simulate_trade_1_2.train.csv"
OUTPUT_FILE_TEST_SIMULATE = "simulate_trade_1_2.test.csv"


# Load data
logger.info("Load data")

df_origin = app_s3.read_dataframe(INPUT_FILE_ORIGIN, index_col=0)
df_train_predict = app_s3.read_dataframe(INPUT_FILE_TRAIN_PREDICT, index_col=0)
df_test_predict = app_s3.read_dataframe(INPUT_FILE_TEST_PREDICT, index_col=0)


# Simulate trade
logger.info("Simulate trade")


def simulate_trade(df):
    df["buy_signal"] = ""
    df["buy_result"] = ""
    df["profit"] = 0.0

    for id in df.index:
        if df.at[id, "predict"] < 4.0:
            df.at[id, "buy_signal"] = "buy"

            if df.at[id, "result"] <= 3:
                df.at[id, "buy_result"] = "win"
                df.at[id, "profit"] = df.at[id, "result_odds_place"] - 1.0
            else:
                df.at[id, "buy_result"] = "lose"
                df.at[id, "profit"] = -1.0


df_train_simulate = pd.concat([df_origin, df_train_predict["predict"]], axis=1).query("not predict.isnull()")
simulate_trade(df_train_simulate)

df_test_simulate = pd.concat([df_origin, df_test_predict["predict"]], axis=1).query("not predict.isnull()")
simulate_trade(df_test_simulate)

app_s3.write_dataframe(df_train_simulate, OUTPUT_FILE_TRAIN_SIMULATE)
app_s3.write_dataframe(df_test_simulate, OUTPUT_FILE_TEST_SIMULATE)


# Report
logger.info("Report")


def report(df):
    report = {}

    report["data_length"] = len(df)

    report["buy_count"] = len(df.query("buy_signal == 'buy'"))
    report["win_count"] = len(df.query("buy_result == 'win'"))
    report["lose_count"] = len(df.query("buy_result == 'lose'"))
    report["win_rate"] = report["win_count"] / report["buy_count"]

    report["profit_total"] = df.query("profit > 0")["profit"].sum()
    report["loss_total"] = df.query("profit < 0")["profit"].sum()
    report["profit_factor"] = report["profit_total"] / abs(report["loss_total"])

    report["profit_average"] = df.query("profit > 0")["profit"].mean()
    report["loss_average"] = df.query("profit < 0")["profit"].mean()
    report["payoff_ratio"] = report["profit_average"] / abs(report["loss_average"])

    return report


report_train = report(df_train_simulate)
report_test = report(df_test_simulate)

logger.info("Report(train): %s" % report_train)
logger.info("Report(test): %s" % report_test)


logger.info("Finish")
