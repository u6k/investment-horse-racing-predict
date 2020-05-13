from flask import Flask

from investment_horse_racing_predict import VERSION
from investment_horse_racing_predict.app_logging import get_logger


logger = get_logger(__name__)


app = Flask(__name__)


@app.route("/api/health")
def health():
    logger.info("#health: start")
    try:

        result = {"version": VERSION}

        return result

    except Exception:
        logger.exception("error")
        return "error", 500
