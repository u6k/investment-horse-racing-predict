from logging import Formatter, getLogger, StreamHandler, DEBUG

logger = getLogger("investment_horse_racing_predict")
formatter = Formatter("%(asctime)-15s - %(levelname)-8s - %(message)s")
handler = StreamHandler()
handler.setLevel(DEBUG)
handler.setFormatter(formatter)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False
