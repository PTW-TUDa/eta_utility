""" Utility functions for connecting to the ENTSO-E Transparency database and for reading data. This connector
does not have the ability to write data.
"""
from __future__ import annotations

import concurrent.futures
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Sized

import pandas as pd
import requests
from lxml import etree
from lxml.builder import E

from eta_utility import get_logger
from eta_utility.connectors.node import NodeEntsoE
from eta_utility.timeseries import df_resample, df_time_slice
from eta_utility.util import dict_search, round_timestamp

if TYPE_CHECKING:
    from typing import Any, Mapping
    from eta_utility.type_hints import AnyNode, Nodes, TimeStep

from .base_classes import BaseSeriesConnection, SubscriptionHandler

log = get_logger("connectors.entso-e")


class ENTSOEConnection(BaseSeriesConnection):
    """
    ENTSOEConnection is a class to download and upload multiple features from and to the ENTSO-E transparency platform
    database as timeseries. The platform contains data about the european electricity markets.

    :param url: Url of the server with scheme (https://transparency.entsoe.eu/)
    :param usr: Username for login to the platform (usually not required - default: None)
    :param pwd: Password for login to the platform (usually not required - default: None)
    :param api_token: Token for API authentication
    :param nodes: Nodes to select in connection
    """

    _PROTOCOL: str = "entsoe"
    API_PATH: str = "/api"

    def __init__(
        self,
        url: str = "https://transparency.entsoe.eu/",
        *,
        api_token: str,
        nodes: Nodes | None = None,
    ) -> None:
        url = url + self.API_PATH
        self._api_token: str = api_token
        super().__init__(url, None, None, nodes=nodes)

        self._node_ids: str | None = None
        self.config = _ConnectionConfiguration()

    @classmethod
    def from_node(
        cls, node: Nodes, *, usr: str | None = None, pwd: str | None = None, **kwargs: Any
    ) -> ENTSOEConnection:
        """Initialize the connection object from an entso-e protocol node object

        :param node: Node to initialize from
        :param usr: Username for login to the platform (usually not required - default: None)
        :param pwd: Password for login to the platform (usually not required - default: None)
        :param api_token: Token for API authentication
        :return: ENTSOEConnection object
        """
        if "api_token" not in kwargs:
            raise AttributeError("Missing required function parameter api_token.")
        api_token = kwargs["api_token"]

        # Make sure nodes is always a set of nodes
        nodes = {node} if not isinstance(node, Sized) else set(node)

        for n in nodes:
            if n.protocol != cls._PROTOCOL:
                raise ValueError(
                    "Tried to initialize ENTSO-E connection with a node that does not specify entso-e as its"
                    "protocol: {}.".format(n.name)
                )

        return cls(next(iter(nodes)).url, api_token=api_token, nodes=node)

    def read(self, nodes: Nodes | None = None) -> pd.DataFrame:
        """
        .. warning::
            Cannot read single values from ENTSO-E transparency platform. Use read_series instead

        :param nodes: List of nodes to read values from
        :return: Pandas DataFrame containing the data read from the connection
        """
        raise NotImplementedError(
            "Cannot read single values from ENTSO-E transparency platform. Use read_series instead"
        )

    def write(self, values: Mapping[AnyNode, Mapping[datetime, Any]], time_interval: timedelta | None = None) -> None:
        """
        .. warning::
            Cannot write to ENTSO-E transparency platform.

        :param values: Dictionary of nodes and data to write. {node: value}
        :param time_interval: Interval between datapoints (i.e. between "From" and "To" in EnEffCo Upload), default 1s
        """
        raise NotImplementedError("Cannot write to ENTSO-E transparency platform.")

    def subscribe(self, handler: SubscriptionHandler, nodes: Nodes | None = None, interval: TimeStep = 1) -> None:
        """Subscribe to nodes and call handler when new data is available. This will return only the
        last available values.

        :param handler: SubscriptionHandler object with a push method that accepts node, value pairs
        :param interval: interval for receiving new data. It is interpreted as seconds when given as an integer.
        :param nodes: identifiers for the nodes to subscribe to
        """
        self.subscribe_series(handler=handler, req_interval=1, nodes=nodes, interval=interval, data_interval=interval)

    def _handle_xml(self, xml_content: bytes) -> pd.DataFrame:
        """Transform XML data from request response into DataFrame.

        :param xml_content: XML data
        :return: DataFrame
        """
        parser = etree.XMLParser(load_dtd=False, ns_clean=True, remove_pis=True)
        xml_data = etree.XML(xml_content, parser)
        ns = xml_data.nsmap
        data = []
        actual_generator_per_type = False

        timeseries = xml_data.findall(".//TimeSeries", namespaces=ns)
        for ts in timeseries:
            col_name = "Price"
            if ts.find(".//MktPSRType", namespaces=ns):
                actual_generator_per_type = True
                psr_type = ts.find(".//MktPSRType", namespaces=ns).find("psrType", namespaces=ns).text
                col_name = dict_search(self.config.psr_types, psr_type)

            # contains the data points
            period = ts.find(".//Period", namespaces=ns)

            # datetime range of the data points
            time_interval = period.find(".//timeInterval", namespaces=ns).getchildren()
            resolution = period.find(".//resolution", namespaces=ns).text[2:4]  # truncating string PR60M
            datetime_range = pd.date_range(
                datetime.strptime(time_interval[0].text, "%Y-%m-%dT%H:%MZ"),
                datetime.strptime(time_interval[1].text, "%Y-%m-%dT%H:%MZ"),
                freq=resolution + "min",
                closed="left",
            )

            points = period.findall(".//Point", namespaces=ns)
            ts_data = [point.getchildren()[-1].text for point in points]

            s = pd.Series(data=ts_data, index=datetime_range, name=col_name)
            data.append(s)

        if actual_generator_per_type:
            return pd.DataFrame(data).T.astype(int)
        return pd.concat(data).astype(float).to_frame()

    def read_series(
        self, from_time: datetime, to_time: datetime, nodes: Nodes | None = None, interval: TimeStep = 1, **kwargs: Any
    ) -> pd.DataFrame:
        """Download timeseries data from the ENTSO-E Database

        :param nodes: List of nodes to read values from
        :param from_time: Starting time to begin reading (included in output)
        :param to_time: Time to stop reading at (not included in output)
        :param interval: interval between time steps. It is interpreted as seconds if given as integer.
        :return: Pandas DataFrame containing the data read from the connection
        """
        nodes = self._validate_nodes(nodes)
        interval = interval if isinstance(interval, timedelta) else timedelta(seconds=interval)

        def read_node(node: NodeEntsoE) -> pd.DataFrame:
            params = self.config.create_params(node, from_time, to_time)

            result = self._raw_request(params)
            df = self._handle_xml(result.content)

            df_res = df_resample(df, interval, missing_data="fillna")
            df_slice = df_time_slice(df_res, from_time, to_time)

            df_slice.columns = [f"{node.name}_{col}" for col in df_slice.columns]

            return df_slice

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(read_node, nodes)

        values = pd.concat(results, axis=1, sort=False)
        return values

    def subscribe_series(
        self,
        handler: SubscriptionHandler,
        req_interval: TimeStep,
        offset: TimeStep | None = None,
        nodes: Nodes | None = None,
        interval: TimeStep = 1,
        data_interval: TimeStep = 1,
        **kwargs: Any,
    ) -> None:
        """
        .. warning::
            Not implemented: Cannot subscribe to data from the ENTSO-E transparency platform.

        :param handler: SubscriptionHandler object with a push method that accepts node, value pairs
        :param req_interval: Duration covered by requested data (time interval). Interpreted as seconds if given as int
        :param offset: Offset from datetime.now from which to start requesting data (time interval).
                       Interpreted as seconds if given as int. Use negative values to go to past timestamps.
        :param data_interval: Time interval between values in returned data. Interpreted as seconds if given as int.
        :param interval: interval (between requests) for receiving new data.
                         It it interpreted as seconds when given as an integer.
        :param nodes: identifiers for the nodes to subscribe to
        """
        raise NotImplementedError("Cannot subscribe to data from the ENTSO-E transparency platform.")

    def close_sub(self) -> None:
        """
        .. warning::
            Not implemented: Cannot subscribe to data from the ENTSO-E transparency platform.
        """
        raise NotImplementedError("Cannot subscribe to data from the ENTSO-E transparency platform.")

    def _raw_request(self, params: Mapping[str, str], **kwargs: Mapping[str, Any]) -> requests.Response:
        """Perform ENTSO-E request and handle possibly resulting errors.

        :param params: Parameters to identify the endpoint
        :param kwargs: Additional arguments for the request.
        :return: request response
        """

        # Prepare the basic request for usage in the requests.
        headers = {"Content-Type": "application/xml", "SECURITY_TOKEN": self._api_token}

        xml = self.config.xml_head()
        for param, val in params.items():
            xml.append(self.config.xml_param(param, val))

        response = requests.post(self.url, data=etree.tostring(xml), headers=headers, **kwargs)  # type: ignore

        # Check for request errors
        if response.status_code != 200:
            e_code = 000
            e_text = "No Message Text"
            if response.status_code == 400:
                try:
                    parser = etree.XMLParser(load_dtd=False, ns_clean=True, remove_pis=True)

                    e_msg = etree.XML(response.content, parser)
                    ns = e_msg.nsmap
                    e_code = e_msg.find(".//Reason", namespaces=ns).find("code", namespaces=ns).text
                    e_text = e_msg.find(".//Reason", namespaces=ns).find("text", namespaces=ns).text
                except Exception:
                    pass
            else:
                try:
                    e_text = etree.HTML(response.content).find("body").text
                except Exception:
                    pass

            error = f"ENTSO-E Error {response.status_code} ({e_code}: {e_text})"

            if response.status_code == 401:
                error = f"{error}: Access Forbidden, Invalid access token"
            elif response.status_code == 404:
                error = f"{error}: Endpoint not found '{self.url}'"
            elif response.status_code == 500:
                error = f"{error}: Server is unavailable"

            raise ConnectionError(error)

        return response


class _ConnectionConfiguration:
    """Auxiliary class to configure the parameters for establish connection to ENTSO-E API.

    Currently, the connection class only supports two types of data requests through the method read_series, they are:
    **Energy price day ahead** and **Actual energy generation per type**. All the data requests available are listed in
    the _doc_type class attribute, but each of them contains a mandatory list of parameters to establish the connection,
    which can be seemed in the ENTSO-E documentation_.

    .. _documentation: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html
    """

    #: XML Namespace for the API
    _xmlns: str = "urn:iec62325.351:tc57wg16:451-5:statusrequestdocument:4:0"

    #: bidding zones is a mapping of three letter iso country codes to bidding zones.
    _bidding_zones = {
        "DEU": "10Y1001A1001A83F",
        "DEU-AUT-LUX": "10Y1001A1001A63L",
        "ALB": "10YAL-KESH-----5",
        "AUT": "10YAT-APG------L",
        "BLR": "10Y1001A1001A51S",
        "BEL": "10YBE----------2",
        "BIH": "10YBA-JPCC-----D",
        "BGR": "10YCA-BULGARIA-R",
        "CZE-DEU-SVK": "10YDOM-CZ-DE-SKK",
        "HRV": "10YHR-HEP------M",
        "CYP": "10YCY-1001A0003J",
        "CZE": "10YCZ-CEPS-----N",
        "DEU-LUX": "10Y1001A1001A82H",
        "DNK_west": "10YDK-1--------W",
        "DNK_central": "10YDK-2--------M",
        "EST": "10Y1001A1001A39I",
        "FIN": "10YFI-1--------U",
        "MKD": "10YMK-MEPSO----8",
        "FRA": "10YFR-RTE------C",
        "GB": "17Y0000009369493",
        "GRC": "10YGR-HTSO-----Y",
        "HUN": "10YHU-MAVIR----U",
        "IRL": "10Y1001A1001A59C",
        "ITA_brindisi": "10Y1001A1001A699",
        "ITA_calabria": "10Y1001C--00096J",
        "ITA_central_north": "10Y1001A1001A70O",
        "ITA_central_south": "10Y1001A1001A71M",
        "ITA_foggia": "10Y1001A1001A72K",
        "ITA-GRC": "10Y1001A1001A66F",
        "ITA_malta": "10Y1001A1001A877",
        "ITA_north": "10Y1001A1001A73I",
        "ITA-AUT": "10Y1001A1001A80L",
        "ITA-CHE": "10Y1001A1001A68B",
        "ITA-FRA": "10Y1001A1001A81J",
        "ITA-SVN": "10Y1001A1001A67D",
        "ITA_priolo": "10Y1001A1001A76C",
        "ITA_rossano": "10Y1001A1001A77A",
        "ITA_sardinia": "10Y1001A1001A74G",
        "ITA_sicily": "10Y1001A1001A75E",
        "ITA_south": "10Y1001A1001A788",
        "RUS_kaliningrad": "10Y1001A1001A50U",
        "LVA": "10YLV-1001A00074",
        "LTU": "10YLT-1001A0008Q",
        "LUX": "10YLU-CEGEDEL-NQ",
        "MLT": "10Y1001A1001A93C",
        "MNE": "10YCS-CG-TSO---S",
        "GBR": "10YGB----------A",
        "NLD": "10YNL----------L",
        "NOR_1": "10YNO-1--------2",
        "NOR_2": "10YNO-2--------T",
        "NOR_3": "10YNO-3--------J",
        "NOR_4": "10YNO-4--------9",
        "NOR_5": "10Y1001A1001A48H",
        "POL": "10YPL-AREA-----S",
        "PRT": "10YPT-REN------W",
        "MDA": "10Y1001A1001A990",
        "ROU": "10YRO-TEL------P",
        "RUS": "10Y1001A1001A49F",
        "SWE_1": "10Y1001A1001A44P",
        "SWE_2": "10Y1001A1001A45N",
        "SWE_3": "10Y1001A1001A46L",
        "SWE_4": "10Y1001A1001A47J",
        "SRB": "10YCS-SERBIATSOV",
        "SVK": "10YSK-SEPS-----K",
        "SVN": "10YSI-ELES-----O",
        "ESP": "10YES-REE------0",
        "SWE": "10YSE-1--------K",
        "CHE": "10YCH-SWISSGRIDZ",
        "TUR": "10YTR-TEIAS----W",
        "UKR": "10Y1001C--00003F",
    }

    _market_agreements = {
        "Daily": "A01",
        "Weekly": "A02",
        "Monthly": "A03",
        "Yearly": "A04",
        "Total": "A05",
        "Long term": "A06",
        "Intraday": "A07",
        "Hourly": "A13",
    }

    _auction_types = {
        "Implicit": "A01",
        "Explicit": "A02",
    }

    _auction_categories = {
        "Base": "A01",
        "Peak": "A02",
        "Off Peak": "A03",
        "Hourly": "A04",
    }

    _psr_types = {
        "Mixed": "A03",
        "Generation": "A04",
        "Load": "A05",
        "Biomass": "B01",
        "Fossil Brown coal/Lignite": "B02",
        "Fossil Coal-derived gas": "B03",
        "Fossil Gas": "B04",
        "Fossil Hard coal": "B05",
        "Fossil Oil": "B06",
        "Fossil Oil shale": "B07",
        "Fossil Peat": "B08",
        "Geothermal": "B09",
        "Hydro Pumped Storage": "B10",
        "Hydro Run-of-river and poundage": "B11",
        "Hydro Water Reservoir": "B12",
        "Marine": "B13",
        "Nuclear": "B14",
        "Other renewable": "B15",
        "Solar": "B16",
        "Waste": "B17",
        "Wind Offshore": "B18",
        "Wind Onshore": "B19",
        "Other": "B20",
        "AC Link": "B21",
        "DC Link": "B22",
        "Substation": "B23",
        "Transformer": "B24",
    }

    _business_types = {
        "General Capacity Information": "A25",
        "Already allocated capacity (AAC)": "A29",
        "Requested capacity (without price)": "A43",
        "System Operator redispatching": "A46",
        "Planned maintenance": "A53",
        "Unplanned outage": "A54",
        "Internal redispatch": "A85",
        "Frequency containment reserve": "A95",
        "Automatic frequency restoration reserve": "A96",
        "Manual frequency restoration reserve": "A97",
        "Replacement reserve": "A98",
        "Interconnector network evolution": "B01",
        "Interconnector network dismantling": "B02",
        "Counter trade": "B03",
        "Congestion costs": "B04",
        "Capacity allocated (including price)": "B05",
        "Auction revenue": "B07",
        "Total nominated capacity": "B08",
        "Net position": "B09",
        "Congestion income": "B10",
        "Production unit": "B11",
        "Area Control Error": "B33",
        "Procured capacity": "B95",
        "Shared Balancing Reserve Capacity": "C22",
        "Share of reserve capacity": "C23",
        "Actual reserve capacity": "C24",
    }

    _process_types = {
        "Day ahead": "A01",
        "Intra day incremental": "A02",
        "Realised": "A16",
        "Intraday total": "A18",
        "Week ahead": "A31",
        "Month ahead": "A32",
        "Year ahead": "A33",
        "Synchronisation process": "A39",
        "Intraday process": "A40",
        "Replacement reserve": "A46",
        "Manual frequency restoration reserve": "A47",
        "Automatic frequency restoration reserve": "A51",
        "Frequency containment reserve": "A52",
        "Frequency restoration reserve": "A56",
    }

    _doc_states = {
        "Intermediate": "A01",
        "Final": "A02",
        "Active": "A05",
        "Cancelled": "A09",
        "Withdrawn": "A13",
        "Estimated": "X01",
    }

    _doc_types = {
        "FinalisedSchedule": "A09",
        "AggregatedEnergyDataReport": "A11",
        "AcquiringSystemOperatorReserveSchedule": "A15",
        "Bid": "A24",
        "AllocationResult": "A25",
        "Capacity": "A26",
        "AgreedCapacity": "A31",
        "ReserveAllocationResult": "A38",
        "Price": "A44",
        "EstimatedNetTransferCapacity": "A61",
        "RedispatchNotice": "A63",
        "SystemTotalLoad": "A65",
        "InstalledGenerationPerType": "A68",
        "WindAndSolarForecast": "A69",
        "LoadForecastMargin": "A70",
        "GenerationForecast": "A71",
        "ReservoirFillingInformation": "A72",
        "ActualGeneration": "A73",
        "WindAndSolarGeneration": "A74",
        "ActualGenerationPerType": "A75",
        "LoadUnavailability": "A76",
        "ProductionUnavailability": "A77",
        "TransmissionUnavailability": "A78",
        "OffshoreGridInfrastructureUnavailability": "A79",
        "GenerationUnavailability": "A80",
        "ContractedReserves": "A81",
        "AcceptedOffers": "A82",
        "ActivatedBalancingQuantities": "A83",
        "ActivatedBalancingPrices": "A84",
        "ImbalancePrices": "A85",
        "ImbalanceVolume": "A86",
        "FinancialSituation": "A87",
        "CrossBorderBalancing": "A88",
        "ContractedReservePrices": "A89",
        "InterconnectionNetworkExpansion": "A90",
        "CounterTradeNotice": "A91",
        "CongestionCosts": "A92",
        "DCLinkCapacity": "A93",
        "NonEUAllocations": "A94",
        "Configuration": "A95",
        "FlowBasedAllocations": "B11",
    }

    def create_params(self, node: NodeEntsoE, from_time: datetime, to_time: datetime) -> dict[str, str]:
        """Create request parameters object according to API specifications
        Handle configuration paramters for each type of connection

        :param node: ENTSO-E Node
        :param from_time: Starting time
        :param to_time: End time
        :return: Dictionary with parameters
        """
        if node.endpoint not in self._doc_types:
            raise ValueError(f"Unsupported endpoint for ENTSO-E connection: {node.endpoint}.")

        params = {"DocumentType": node.endpoint}
        if node.endpoint == "ActualGenerationPerType":
            params["ProcessType"] = "Realised"
            params["In_Domain"] = node.bidding_zone
            resolution = 15  # 15 minutes interval data

        elif node.endpoint == "Price":
            params["ProcessType"] = "Day ahead"
            params["In_Domain"] = node.bidding_zone
            params["Out_Domain"] = node.bidding_zone
            resolution = 60  # 1 hour interval data

        else:
            raise NotImplementedError(f"Endpoint not available: {node.endpoint}")

        rounded_from_time = round_timestamp(from_time, interval=resolution * 60, ensure_tz=False)
        rounded_to_time = round_timestamp(to_time, interval=resolution * 60, ensure_tz=False)
        params["TimeInterval"] = (
            f"{rounded_from_time.isoformat(sep='T', timespec='minutes')}Z/"
            f"{rounded_to_time.isoformat(sep='T', timespec='minutes')}Z"
        )
        return params

    def xml_head(self) -> etree.ElementTree:
        """Create header of the xml data for the POST method.

        :return: tree of elements with the pre-defined values for the request
        """
        now = datetime.utcnow()
        # Prepare XML Header data
        data = E("StatusRequest_MarketDocument", xmlns=self._xmlns)
        data.append(E("mRID", f"Request_{now.isoformat(sep='T', timespec='seconds')}"))
        data.append(E("type", "A59"))
        data.append(E("sender_MarketParticipant.mRID", "10X1001A1001A450", codingScheme="A01"))
        data.append(E("sender_MarketParticipant.marketRole.type", "A07"))
        data.append(E("receiver_MarketParticipant.mRID", "10X1001A1001A450", codingScheme="A01"))
        data.append(E("receiver_MarketParticipant.marketRole.type", "A32"))
        data.append(E("createdDateTime", f"{now.isoformat(sep='T', timespec='seconds')}Z"))

        return data

    def xml_param(self, parameter: str, value: str) -> etree.Element:
        """Map parameters to request values for the xml document.

        :return: tree with parameters
        """
        if parameter == "Contract_MarketAgreement.Type" or parameter == "Type_MarketAgreement.Type":
            value = self._market_agreements[value]
        elif parameter == "Auction.Type":
            value = self._auction_types[value]
        elif parameter == "Auction.Category":
            value = self._auction_categories[value]
        elif parameter == "PsrType":
            value = self._psr_types[value]
        elif parameter == "BusinessType":
            value = self._business_types[value]
        elif parameter == "ProcessType":
            value = self._process_types[value]
        elif parameter == "DocStatus":
            value = self._doc_states[value]
        elif parameter == "DocumentType":
            value = self._doc_types[value]
        elif parameter in {"In_Domain", "Out_Domain"}:
            value = self._bidding_zones[value]

        return E("AttributeInstanceComponent", E("attribute", parameter), E("attributeValue", value))

    @property
    def psr_types(self) -> dict[str, str]:
        return self._psr_types

    @property
    def doc_types(self) -> dict[str, str]:
        return self._doc_types
