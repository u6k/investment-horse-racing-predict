import json

from investment_horse_racing_predict import flask, VERSION


class TestFlask:
    def setUp(self):
        self.app = flask.app.test_client()

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

    def test_predict_vote(self):
        # Setup
        req = {
            "race_id": "1906050204",
            "asset": 1000000,
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
        assert result_data["win"]["vote_cost"] > 0

    def test_predict_no_vote(self):
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
