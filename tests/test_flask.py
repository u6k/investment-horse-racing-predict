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
