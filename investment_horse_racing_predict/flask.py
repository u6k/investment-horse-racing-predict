import os
from flask import Flask, request, g
import psycopg2
from psycopg2.extras import DictCursor
from queue import Queue
import time
import functools
import pandas as pd

from investment_horse_racing_predict import VERSION, main
from investment_horse_racing_predict.app_logging import get_logger


logger = get_logger(__name__)


app = Flask(__name__)


pd.options.display.max_columns = 1000
pd.options.display.show_dimensions = True
pd.options.display.width = 10000


singleQueue = Queue(maxsize=1)


def multiple_control(q):
    def _multiple_control(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            q.put(time.time())
            logger.debug("#multiple_control: start: critical zone")
            result = func(*args, **kwargs)
            logger.debug("#multiple_control: end: critical zone")
            q.get()
            q.task_done()
            return result
        return wrapper
    return _multiple_control


@app.route("/api/health")
def health():
    logger.info("#health: start")
    try:

        result = {"version": VERSION}

        # Check database
        logger.debug("#health: check database")

        with get_crawler_db().cursor() as db_cursor:
            db_cursor.execute("select 1")
            result["database"] = True

        # Check model
        logger.debug("#health: check model")

        model_data = main.load_json_from_url(os.getenv("RESULT_PREDICT_MODEL_URL"))
        result["result_predict_model.algorithm"] = model_data["algorithm"]

        vote_parameters = main.load_json_from_url(os.getenv("VOTE_PREDICT_MODEL_URL"))
        result["vote_predict_model.algorithm"] = vote_parameters["algorithm"]

        return result

    except Exception:
        logger.exception("error")
        return "error", 500


@app.route("/api/predict", methods=["POST"])
@multiple_control(singleQueue)
def predict():
    logger.info("#predict: start")
    try:

        args = request.get_json()
        logger.debug(f"#predict: args={args}")

        race_id = args.get("race_id")
        asset = args.get("asset")
        vote_cost_limit = args.get("vote_cost_limit", 10000)

        result = main.predict(race_id, asset, vote_cost_limit)

        return result

    except Exception:
        logger.exception("error")
        return "error", 500


def get_crawler_db():
    if "crawler_db" not in g:
        g.crawler_db = psycopg2.connect(
            host=os.getenv("CRAWLER_DB_HOST"),
            port=os.getenv("CRAWLER_DB_PORT"),
            dbname=os.getenv("CRAWLER_DB_DATABASE"),
            user=os.getenv("CRAWLER_DB_USERNAME"),
            password=os.getenv("CRAWLER_DB_PASSWORD")
        )
        g.crawler_db.autocommit = False
        g.crawler_db.set_client_encoding("utf-8")
        g.crawler_db.cursor_factory = DictCursor

    return g.crawler_db


@app.teardown_appcontext
def _teardown_db(exc):
    crawler_db = g.pop("crawler_db", None)
    if crawler_db is not None:
        crawler_db.close()
