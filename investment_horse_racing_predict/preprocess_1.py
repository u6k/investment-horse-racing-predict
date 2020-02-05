import logging
import pandas as pd
import os
import psycopg2
import app_s3

logger = logging.getLogger(__name__)

db_conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_DATABASE"),
    user=os.getenv("DB_USERNAME"),
    password=os.getenv("DB_PASSWORD")
)
db_conn.set_client_encoding("utf-8")

sql = """
    select
        r.race_id,
        r.result,
        r.bracket_number,
        r.horse_number,
        r.horse_id,
        r.horse_weight,
        r.horse_weight_diff,
        r.arrival_time,
        r.jockey_id,
        r.jockey_weight,
        r.favorite_order,
        r.trainer_id,
        i.race_round,
        i.start_datetime,
        i.course_type,
        i.course_length,
        i.weather,
        i.course_condition,
        h.gender,
        h.birthday,
        h.coat_color,
        ow.odds as odds_win,
        op.odds_min as odds_place_min,
        op.odds_max as odds_place_max,
        pw.odds as result_odds_win,
        pp.odds as result_odds_place
    from
        race_result as r left outer join race_info as i on
            r.race_id = i.race_id
        left outer join horse as h on
            r.horse_id = h.horse_id
        left outer join odds_win as ow on
            r.race_id = ow.race_id
            and r.horse_number = ow.horse_number
        left outer join odds_place as op on
            r.race_id = op.race_id
            and r.horse_number = op.horse_number
        left outer join race_payoff as pw on
            r.race_id = pw.race_id
            and r.horse_number = pw.horse_number
            and pw.payoff_type = 'win'
        left outer join race_payoff as pp on
            r.race_id = pp.race_id
            and r.horse_number = pp.horse_number
            and pp.payoff_type = 'place'
    where
        i.start_datetime > '2005-01-01'
    order by
        i.start_datetime, i.race_id
"""

df = pd.read_sql(sql, db_conn)
df.info()
df.head()

app_s3.write_dataframe(df, "preprocess_1.csv")
