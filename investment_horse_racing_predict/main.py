import urllib.request
import json
import pandas as pd
import numpy as np

from investment_horse_racing_predict import flask
from investment_horse_racing_predict.app_logging import get_logger


logger = get_logger(__name__)



def predict(race_id, asset, vote_cost_limit):
    logger.info(f"#predict: start: race_id={race_id}, asset={asset}, vote_cost_limit={vote_cost_limit}")

    df = join_crawled_data(race_id)
    logger.debug(df.info()) # TODO
    logger.debug(df) # TODO





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
            inner join race_info as i on
                d.race_id = i.race_id
                and i.start_datetime >= '{start_date}'
                and i.start_datetime <= '{end_date}'
            inner join race_result as r on
                d.race_id = r.race_id
                and d.horse_number = r.horse_number
            inner join odds_win as o on
                d.race_id = r.race_id
                and d.horse_number = o.horse_number
            inner join horse as h on
                d.horse_id = h.horse_id
            inner join jockey as j on
                d.jockey_id = j.jockey_id
            inner join trainer as t on
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









def load_json_from_url(url):
    with urllib.request.urlopen(url) as response:
        data = json.load(response)

    return data
