import pathlib


class Config:
    ENEFFCO_USER = ""
    ENEFFCO_PW = ""
    ENEFFCO_URL = ""
    ENEFFCO_POSTMAN_TOKEN = ""

    CSV_OUTPUT_FILE = pathlib.Path(__file__).parent / "test_resources/test_output.csv"
    FMU_FILE = pathlib.Path(__file__).parent / "test_resources/etax/environment/damped_oscillator.fmu"
    LIVE_CONNECT_CONFIG = pathlib.Path(__file__).parent / "test_resources/config_live_connect.json"
    EXCEL_NODES_FILE = pathlib.Path(__file__).parent / "test_resources/test_excel_node_list.xls"
    EXCEL_NODES_SHEET = "Sheet1"
