import logging
import pandas as pd
import os
import psycopg2

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
        r.odds,
        r.trainer_id,
        i.race_round,
        i.start_datetime,
        i.course_type,
        i.course_length,
        i.weather,
        i.course_condition,
        h.gender,
        h.birthday,
        h.coat_color
    from
        race_result as r left outer join race_info as i on r.race_id = i.race_id
        left outer join horse as h on r.horse_id = h.horse_id
"""

df = pd.read_sql(sql, db_conn)
