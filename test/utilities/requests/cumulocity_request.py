import json
import pathlib

from requests import Response as _Response


class Response(_Response):
    def __init__(self, json_data=None, status_code=400):
        super().__init__()
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data

    def text(self):
        pass


def request(method, url, **kwargs):
    if url == "":
        return Response(status_code=200)
    else:
        try:
            measurement_id = url.split("source=", 1)[1].split("&valueFragmentSeries=", 1)[0]
            value_fragment_series = url.split("&valueFragmentSeries=", 1)[1].split("&pageSize=", 1)[0]
            if "currentPage" in url:
                current_page = url.split("currentPage=", 1)[1]
            else:
                current_page = "1"
        except IndexError:
            measurement_id = ""

    with open(pathlib.Path(__file__).parent / "cumulocity_sample_data.json") as f:
        data = json.load(f)

    if method == "GET":
        if measurement_id == "1234" and value_fragment_series == "P" and current_page == "1":
            # Return data for id: 1234 and series: P
            return Response(data, 200)

        elif measurement_id == "1235" and value_fragment_series == "P" and current_page == "1":
            # Return data for id: 1234 and series: P
            return Response(data, 200)

        elif current_page == "2":
            return Response({"measurements": []}, status_code=200)

        else:
            return Response(status_code=404)
