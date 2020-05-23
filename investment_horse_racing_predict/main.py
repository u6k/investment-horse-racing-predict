import urllib.request
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle
import base64
import math

from investment_horse_racing_predict import flask
from investment_horse_racing_predict.app_logging import get_logger


logger = get_logger(__name__)


def predict(race_id, asset, vote_cost_limit):
    logger.info(f"#predict: start: race_id={race_id}, asset={asset}, vote_cost_limit={vote_cost_limit}")

    df_join = join_crawled_data(race_id)

    df_score, df_score_horse, df_score_jockey, df_score_trainer = calc_horse_jockey_trainer_score(df_join)

    df_join = pd.merge(df_join, df_score_horse[["race_id", "horse_id", "horse_score"]], on=["race_id", "horse_id"], how="left")
    df_join = pd.merge(df_join, df_score_jockey[["race_id", "jockey_id", "jockey_score"]], on=["race_id", "jockey_id"], how="left")
    df_join = pd.merge(df_join, df_score_trainer[["race_id", "trainer_id", "trainer_score"]], on=["race_id", "trainer_id"], how="left")

    df_all = merge_past_race(df_join)

    df, df_data, df_query, df_label = split_data_query_label(df_all, race_id)

    horse_numbers, df_result, model_data = predict_result(df, df_data)

    vote_cost, vote_parameters = calc_vote_cost(asset, vote_cost_limit, race_id, horse_numbers[0])

    result = {
        "race_id": race_id,
        "vote_parameters": vote_parameters,
        "predict_parameters": {
            "algorithm": model_data["algorithm"],
        },
        "win": {
            "horse_number": horse_numbers[0],
            "odds": vote_parameters["parameters"]["odds"],
            "vote_cost": vote_cost,
        },
    }
    logger.debug(f"#predict: result={result}")

    df.to_csv(f"/var/dataframe/df.{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv")
    df_result.to_csv(f"/var/dataframe/result.{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv")
    logger.debug("#predict: csv saved")

    return result


def join_crawled_data(race_id, data_period=365):
    logger.info(f"#join_crawled_data: start: race_id={race_id}")

    logger.debug("#join_crawled_data: read sql")

    with flask.get_crawler_db() as db_conn:
        sql = f"select start_datetime from race_info where race_id='{race_id}'"

        df_tmp = pd.read_sql(sql=sql, con=db_conn)
        end_date = df_tmp["start_datetime"].values[0]
        start_date = end_date - np.timedelta64(data_period, "D")

        logger.debug(f"#join_crawled_data: start_date={start_date}, end_date={end_date}")

        sql = f"""select
            d.race_id,
            d.bracket_number,
            d.horse_number,
            d.horse_id,
            d.trainer_id,
            d.horse_weight,
            d.horse_weight_diff,
            d.jockey_id,
            d.jockey_weight,
            d.prize_total_money,
            i.race_round,
            i.start_datetime,
            i.place_name,
            i.course_type,
            i.course_length,
            i.weather,
            i.course_condition,
            r.result,
            r.arrival_time,
            r.favorite_order,
            o.odds as odds_win,
            h.gender as gender_horse,
            h.birthday as birthday_horse,
            h.coat_color,
            j.birthday as birthday_jockey,
            j.first_licensing_year as first_licensing_year_jockey,
            t.birthday as birthday_trainer,
            t.first_licensing_year as first_licensing_year_trainer
        from
            race_denma as d
            left join race_info as i on
                d.race_id = i.race_id
                and i.start_datetime >= '{start_date}'
                and i.start_datetime <= '{end_date}'
            left join race_result as r on
                d.race_id = r.race_id
                and d.horse_number = r.horse_number
            left join odds_win as o on
                d.race_id = o.race_id
                and d.horse_number = o.horse_number
            left join horse as h on
                d.horse_id = h.horse_id
            left join jockey as j on
                d.jockey_id = j.jockey_id
            left join trainer as t on
                d.trainer_id = t.trainer_id
        order by i.start_datetime, d.horse_number"""

        df_tmp = pd.read_sql(sql=sql, con=db_conn)

    logger.debug("#join_crawled_data: get dummies")

    df_tmp.replace({
        "course_type": {"^ダート.*": 1, "^芝.*": 2, "^障害.*": 3},
        "place_name": {".*京都.*": 1, ".*中山.*": 2, ".*小倉.*": 3, ".*阪神.*": 4, ".*東京.*": 5, ".*札幌.*": 6, ".*函館.*": 7, ".*新潟.*": 8, ".*中京.*": 9, ".*福島.*": 10},
    }, regex=True, inplace=True)

    df_tmp.replace({
        "weather": {'晴': 1, '曇': 2, '雨': 3, '小雨': 3, '雪': 4, '小雪': 4},
        "course_condition": {'良': 1, '稍重': 2, '重': 3, '不良': 4},
        "gender_horse": {'牡': 1, '牝': 2, 'せん': 3},
        "coat_color": {'青鹿毛': 1, '栗毛': 2, '鹿毛': 3, '芦毛': 4, '黒鹿毛': 5, '青毛': 6, '白毛': 7, '栃栗毛': 8},
    }, inplace=True)

    logger.debug("#join_crawled_data: fillna")

    df_tmp.fillna({
        "result": 21,
        "horse_weight": 999,
        "arrival_time": 600,
        "jockey_weight": 99,
        "favorite_order": 21,
        "course_length": 6000,
        "odds_win": 999,
        "speed": 10,
        "birthday_horse": datetime(1900, 1, 1),
        "birthday_jockey": datetime(1900, 1, 1),
        "birthday_trainer": datetime(1900, 1, 1),
    }, inplace=True)
    df_tmp.fillna(0, inplace=True)

    logger.debug("#join_crawled_data: calc")

    df_tmp["speed"] = df_tmp["course_length"] / df_tmp["arrival_time"]

    df_tmp["birth_age_horse"] = (df_tmp["start_datetime"] - df_tmp["birthday_horse"]) / np.timedelta64(1, "D")
    df_tmp["birth_age_jockey"] = (df_tmp["start_datetime"] - df_tmp["birthday_jockey"]) / np.timedelta64(1, "D")
    df_tmp["birth_age_trainer"] = (df_tmp["start_datetime"] - df_tmp["birthday_trainer"]) / np.timedelta64(1, "D")
    df_tmp["licensing_age_jockey"] = df_tmp["start_datetime"].dt.year - df_tmp["first_licensing_year_jockey"]
    df_tmp["licensing_age_trainer"] = df_tmp["start_datetime"].dt.year - df_tmp["first_licensing_year_trainer"]

    df_tmp.drop(["birthday_horse", "birthday_jockey", "birthday_trainer", "first_licensing_year_jockey", "first_licensing_year_trainer"], axis=1, inplace=True)

    return df_tmp


def calc_horse_jockey_trainer_score(df_arg):
    logger.info("#calc_horse_jockey_trainer_score: start")

    logger.debug("#calc_horse_jockey_trainer_score: merge, calc score")

    df_score = df_arg[["race_id", "start_datetime", "horse_id", "jockey_id", "trainer_id", "result"]]
    df_score["score"] = 1 / df_score["result"]

    logger.debug("#calc_horse_jockey_trainer_score: calc horse score")

    df_score_horse = df_score[["race_id", "start_datetime", "horse_id", "score"]].sort_values(["horse_id", "start_datetime"])
    df_score_horse = df_score_horse.groupby(["race_id", "start_datetime", "horse_id"])[["score"]].sum()
    df_score_horse = df_score_horse.groupby("horse_id").rolling(10000, min_periods=1)[["score"]].sum()
    df_score_horse.index = df_score_horse.index.droplevel(0)
    df_score_horse.reset_index(inplace=True)
    df_score_horse = pd.merge(df_score_horse, df_score_horse[["horse_id", "score"]].shift(1), left_index=True, right_index=True)
    df_score_horse.loc[df_score_horse["horse_id_x"] != df_score_horse["horse_id_y"], "score_y"] = 0.0
    df_score_horse.drop(["score_x", "horse_id_y"], axis=1, inplace=True)
    df_score_horse.rename(columns={"horse_id_x": "horse_id", "score_y": "horse_score"}, inplace=True)

    logger.debug("#calc_horse_jockey_trainer_score: calc jockey score")

    df_score_jockey = df_score[["race_id", "start_datetime", "jockey_id", "score"]].sort_values(["jockey_id", "start_datetime"])
    df_score_jockey = df_score_jockey.groupby(["race_id", "start_datetime", "jockey_id"])[["score"]].sum()
    df_score_jockey = df_score_jockey.groupby("jockey_id").rolling(10000, min_periods=1)[["score"]].sum()
    df_score_jockey.index = df_score_jockey.index.droplevel(0)
    df_score_jockey.reset_index(inplace=True)
    df_score_jockey = pd.merge(df_score_jockey, df_score_jockey[["jockey_id", "score"]].shift(1), left_index=True, right_index=True)
    df_score_jockey.loc[df_score_jockey["jockey_id_x"] != df_score_jockey["jockey_id_y"], "score_y"] = 0.0
    df_score_jockey.drop(["score_x", "jockey_id_y"], axis=1, inplace=True)
    df_score_jockey.rename(columns={"jockey_id_x": "jockey_id", "score_y": "jockey_score"}, inplace=True)

    logger.debug("#calc_horse_jockey_trainer_score: calc trainer score")

    df_score_trainer = df_score[["race_id", "start_datetime", "trainer_id", "score"]].sort_values(["trainer_id", "start_datetime"])
    df_score_trainer = df_score_trainer.groupby(["race_id", "start_datetime", "trainer_id"])[["score"]].sum()
    df_score_trainer = df_score_trainer.groupby("trainer_id").rolling(10000, min_periods=1)[["score"]].sum()
    df_score_trainer.index = df_score_trainer.index.droplevel(0)
    df_score_trainer.reset_index(inplace=True)
    df_score_trainer = pd.merge(df_score_trainer, df_score_trainer[["trainer_id", "score"]].shift(1), left_index=True, right_index=True)
    df_score_trainer.loc[df_score_trainer["trainer_id_x"] != df_score_trainer["trainer_id_y"], "score_y"] = 0.0
    df_score_trainer.drop(["score_x", "trainer_id_y"], axis=1, inplace=True)
    df_score_trainer.rename(columns={"trainer_id_x": "trainer_id", "score_y": "trainer_score"}, inplace=True)

    return df_score, df_score_horse, df_score_jockey, df_score_trainer


def merge_past_race(df_arg, past_len=3):
    logger.info("#merge_past_race: start")

    logger.debug("#merge_past_race: merge")

    df_all = df_arg.sort_values(["horse_id", "start_datetime"]).reset_index(drop=True)
    df_tmp = df_all.copy()

    for shift_i in range(1, past_len+1):
        df_all = pd.merge(df_all, df_tmp.shift(shift_i), left_index=True, right_index=True, suffixes=("", f"_{shift_i}"))

    logger.debug("#merge_past_race: set none")

    for shift_i in range(1, past_len+1):
        for col in df_all.columns:
            if col.endswith(f"_{shift_i}"):
                df_all.loc[df_all["horse_id"] != df_all[f"horse_id_{shift_i}"], col] = None

    logger.debug("#merge_past_race: drop")

    for shift_i in range(1, 4):
        df_all.drop([
            f"race_id_{shift_i}",
            f"horse_id_{shift_i}",
            f"jockey_id_{shift_i}",
            f"trainer_id_{shift_i}",
            f"start_datetime_{shift_i}"
        ], axis=1, inplace=True)

    logger.debug("#merge_past_race: fillna")

    df_all.fillna({
        "result": 21,
        "horse_weight": 999,
        "arrival_time": 600,
        "jockey_weight": 99,
        "favorite_order": 21,
        "course_length": 6000,
        "odds_win": 999,
        "speed": 10,
    }, inplace=True)

    for shift_i in range(1, 4):
        df_all.fillna({
            f"result_{shift_i}": 21,
            f"horse_weight_{shift_i}": 999,
            f"arrival_time_{shift_i}": 600,
            f"jockey_weight_{shift_i}": 99,
            f"favorite_order_{shift_i}": 21,
            f"course_length_{shift_i}": 6000,
            f"odds_win_{shift_i}": 999,
            f"speed_{shift_i}": 10,
        }, inplace=True)

    df_all.fillna(0, inplace=True)

    return df_all


def split_data_query_label(df, race_id):
    logger.info(f"#split_data_query_label: start: race_id={race_id}")

    df_tmp = df.drop([
        "favorite_order",
        "odds_win",
        "arrival_time", "speed",
        "horse_id", "jockey_id", "trainer_id"
    ], axis=1)

    df_tmp = df_tmp.query(f"race_id=='{race_id}'")
    df_tmp.sort_values(["race_id", "horse_number"], inplace=True)

    df_tmp_label = df_tmp[["result"]]
    df_tmp_label["label"] = df_tmp_label["result"].apply(lambda r: 3 if r == 1 else (2 if r == 2 else (1 if r == 3 else 0)))
    df_tmp_query = pd.DataFrame(df_tmp.groupby("race_id").size())
    df_tmp_data = df_tmp.drop(["race_id", "start_datetime", "result"], axis=1)

    return df_tmp, df_tmp_data, df_tmp_query, df_tmp_label


def predict_result(df_arg, df_data):
    logger.info("#predict_result: start")

    logger.debug("#predict_result: load model")

    model_data = load_json_from_url(os.getenv("RESULT_PREDICT_MODEL_URL"))
    lgb_model = pickle.loads(base64.b64decode(model_data["model"].encode()))
    logger.debug(f"#predict_result: algorithm={model_data['algorithm']}")

    logger.debug("#predict_result: predict")

    df_result = df_arg[["race_id", "horse_number", "start_datetime", "race_round"]]
    df_result["pred"] = lgb_model.predict(df_data, num_iteration=lgb_model.best_iteration)

    logger.debug("#predict_result: calc rank")

    for race_id, df_chunk in df_result.groupby("race_id"):
        for rank, index in enumerate(df_chunk.sort_values("pred", ascending=False).index):
            df_result.at[index, "pred_result"] = rank + 1
    df_result.sort_values("pred_result", inplace=True)
    horse_numbers = df_result["horse_number"].values.astype(int).tolist()

    logger.debug(f"#predict_result: horse_numbers={horse_numbers}")

    return horse_numbers, df_result, model_data


def calc_vote_cost(asset, vote_cost_limit, race_id, horse_number):
    logger.info(f"#calc_vote_cost: start: asset={asset}, vote_cost_limit={vote_cost_limit}, race_id={race_id}, horse_number={horse_number}")

    logger.debug("#calc_vote_cost: load parameters")

    vote_parameters = load_json_from_url(os.getenv("VOTE_PREDICT_MODEL_URL"))
    logger.debug(f"#calc_vote_cost: vote_parameters={vote_parameters}")

    hit_rate = vote_parameters["parameters"]["hit_rate"]
    kelly_coefficient = vote_parameters["parameters"]["kelly_coefficient"]

    logger.debug("#calc_vote_cost: load odds")

    with flask.get_crawler_db() as db_conn:
        sql = f"select odds from odds_win where race_id = '{race_id}' and horse_number = '{horse_number}'"

        df = pd.read_sql(sql=sql, con=db_conn)
        odds = df["odds"].values[0]
        logger.debug(f"#calc_vote_cost: odds={odds}")

    logger.debug("#calc_vote_cost: calc")

    if odds > 1.0:
        kelly = (hit_rate * odds - 1.0) / (odds - 1.0)
    else:
        kelly = 0.0

    if kelly > 0.0:
        vote_cost = math.floor(asset * kelly * kelly_coefficient / 100.0) * 100
        if vote_cost > vote_cost_limit:
            vote_cost = vote_cost_limit
    else:
        vote_cost = 0

    vote_parameters["parameters"]["odds"] = odds
    vote_parameters["parameters"]["kelly"] = kelly

    logger.debug(f"#calc_vote_cost: vote_cost={vote_cost}, vote_parameters={vote_parameters}")

    return vote_cost, vote_parameters


def load_json_from_url(url):
    with urllib.request.urlopen(url) as response:
        data = json.load(response)

    return data
