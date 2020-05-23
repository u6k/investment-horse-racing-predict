import json

from investment_horse_racing_predict import flask, VERSION


class TestFlask:
    def setUp(self):
        self.app = flask.app.test_client()

        # Setup crawler_db
        with flask.get_crawler_db_without_flask() as db_conn:
            with db_conn.cursor() as db_cursor:
                db_cursor.execute("delete from race_info")
                db_cursor.execute("delete from race_denma")
                db_cursor.execute("delete from race_payoff")
                db_cursor.execute("delete from race_result")
                db_cursor.execute("delete from odds_win")
                db_cursor.execute("delete from odds_place")
                db_cursor.execute("delete from horse")
                db_cursor.execute("delete from jockey")
                db_cursor.execute("delete from trainer")

                with open("tests/data/crawler_db/race_info.csv") as f:
                    db_cursor.copy_from(f, "race_info", null="")
                with open("tests/data/crawler_db/race_denma.csv") as f:
                    db_cursor.copy_from(f, "race_denma", null="")
                with open("tests/data/crawler_db/race_payoff.csv") as f:
                    db_cursor.copy_from(f, "race_payoff", null="")
                with open("tests/data/crawler_db/race_result.csv") as f:
                    db_cursor.copy_from(f, "race_result", null="")
                with open("tests/data/crawler_db/odds_win.csv") as f:
                    db_cursor.copy_from(f, "odds_win", null="")
                with open("tests/data/crawler_db/odds_place.csv") as f:
                    db_cursor.copy_from(f, "odds_place", null="")
                with open("tests/data/crawler_db/horse.csv") as f:
                    db_cursor.copy_from(f, "horse", null="")
                with open("tests/data/crawler_db/jockey.csv") as f:
                    db_cursor.copy_from(f, "jockey", null="")
                with open("tests/data/crawler_db/trainer.csv") as f:
                    db_cursor.copy_from(f, "trainer", null="")

            db_conn.commit()

    def test_health(self):
        # Execute
        result = self.app.get("/api/health")

        # Check
        assert result.status_code == 200

        result_data = json.loads(result.get_data(as_text=True))
        assert result_data["version"] == VERSION
        assert result_data["database"]
        assert result_data["result_predict_model.algorithm"] is not None
        assert result_data["vote_predict_model.algorithm"] is not None

    def test_predict(self):
        # Setup
        req = {
            "race_id": "1909050201",
            "asset": 10000,
        }

        # Execute
        result = self.app.post("/api/predict", json=req)

        # Check
        assert result.status_code == 200

        result_data = json.loads(result.get_data(as_text=True))
        assert result_data["race_id"] == req["race_id"]
        assert type(result_data["vote_parameters"]) == dict
        assert type(result_data["predict_parameters"]) == dict
        assert type(result_data["win"]["horse_number"]) == int
        assert type(result_data["win"]["odds"]) == float
        assert type(result_data["win"]["vote_cost"]) == int
