import json

from config_tests import HOST, PORT
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/Test/PutJson", methods=["PUT"])
def put_json():
    with open("test/test_data.json", "w") as outputfile:
        json.dump(request.json, outputfile)
    return jsonify("OK")


@app.route("/Test/GetJson")
def get_json():
    with open("test/test_data.json", "r") as input:
        output = json.load(input)
    return jsonify(output)


if __name__ == "__main__":
    app.run(HOST, PORT)
