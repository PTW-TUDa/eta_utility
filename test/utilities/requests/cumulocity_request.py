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
        if method == "GET":
            with open(pathlib.Path(__file__).parent / "cumulocity_sample_data.json") as f:
                data = json.load(f)

            if "/measurement/measurements" in url:
                try:
                    device_id = url.split("source=", 1)[1].split("&", 1)[0]
                    if "valueFragmentSeries" in url:
                        fragment = url.split("&valueFragmentSeries=", 1)[1].split("&", 1)[0]
                        assert fragment == "P"
                    else:
                        fragment = ""
                    if "currentPage" in url:
                        current_page = url.split("currentPage=", 1)[1]
                    else:
                        current_page = "1"
                except IndexError:
                    device_id = ""
                if device_id == "1234" and current_page == "1":
                    # Return data for id: 1234 and fragment: P
                    return Response(data, 200)

                elif device_id == "1235" and current_page == "1":
                    # Return data for id: 1234 and fragment: P
                    return Response(data, 200)

                elif current_page == "2":
                    return Response({"measurements": []}, status_code=200)

                else:
                    return Response(status_code=404)
        if method == "POST":
            if "/measurement/measurements" in url:
                payload_true = {
                    "source": {"id": "1234"},
                    "time": "2023-12-06T08:41:26",
                    "type": "power",
                    "Power": {"P": {"unit": "W", "value": 1}},
                }

                if "data" in kwargs.keys():
                    if payload_true == json.loads(kwargs.get("data")):
                        return Response(status_code=200)
                    else:
                        raise Exception(
                            "Payload was incorrect. Payload should be {}, but got {}.".format(
                                payload_true, json.loads(kwargs.get("data"))
                            )
                        )
                else:
                    raise Exception("Request had no payload.")

            elif "/inventory/managedObjects" in url:
                assert "data" in kwargs.keys()
                assert "name" in json.loads(kwargs.get("data")).keys()
                assert "c8y_IsDevice" in json.loads(kwargs.get("data")).keys()
                assert json.loads(kwargs.get("data"))["name"] == "Device1"
                return Response(status_code=200)
