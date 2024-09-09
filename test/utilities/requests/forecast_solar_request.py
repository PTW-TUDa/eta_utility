import json
import pathlib
import re

from requests import Response as _Response
from requests_cache import CachedResponse as _CachedResponse


class Response(_Response):
    def __init__(self, url, json_data=None, status_code=400):
        super().__init__()
        self.url = url
        self.json_data = json_data
        self.status_code = status_code
        self.reason = "MOCK RESPONSE REASON"

    def json(self):
        return self.json_data

    def text(self):
        pass


class CachedResponse(_CachedResponse):
    def __init__(self, url, json_data=None, status_code=400):
        super().__init__()
        self.url = url
        self.json_data = json_data
        self.status_code = status_code
        self.reason = "MOCK RESPONSE REASON"

    def json(self):
        return self.json_data

    def text(self):
        pass


def request(self, method, url: str, *args, **kwargs):
    try:
        endpoint = url.split("https://api.forecast.solar", 1)[1]
        pattern_api_key = r"^/[A-Za-z0-9]{16}/"
        endpoint = re.sub(r"/\d.*", r"", endpoint)  # Remove any trailing parameters
        endpoint = re.sub(pattern_api_key, r"/", endpoint)  # Remove API key from endpoint
    except Exception:
        if "httpbin.org" in url:
            return Response(url, status_code=200)
        endpoint = ""

    # with open(pathlib.Path(__file__).parent / "forecast_solar_sample_data.json") as f:
    with pathlib.Path(pathlib.Path(__file__).parent / "forecast_solar_sample_data.json").open() as f:
        data = json.load(f)

    if method == "GET":
        if endpoint in ["/help", "/check"]:
            # Empty request or check
            return CachedResponse(url, status_code=200)
        if endpoint == "/estimate/watts":
            # Datapoint IDs
            return CachedResponse(url, data, 200)

        if endpoint in ["/clearsky", "/history", "/timewindows", "/weather", "/chart"]:
            raise NotImplementedError(f"Mock request of endpoint: '{endpoint}' is not implemented yet.")

        return CachedResponse(url, status_code=404)

    return CachedResponse(url, status_code=200)
