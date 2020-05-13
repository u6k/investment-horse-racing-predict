import urllib.request
import json

from investment_horse_racing_predict.app_logging import get_logger





logger = get_logger(__name__)






def load_json_from_url(url):
    with urllib.request.urlopen(url) as response:
        data = json.load(response)

    return data
