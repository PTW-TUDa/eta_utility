""" Initiate live connections that automate certain tasks associated with the creation of such connections."""
import pathlib
import types
from contextlib import AbstractContextManager
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type, Union
from urllib.parse import urlparse, urlunparse

from eta_utility import get_logger, json_import
from eta_utility.connectors import (
    Node,
    connections_from_nodes,
    default_schemes,
    name_map_from_node_sequence,
)
from eta_utility.type_hints import Connection, Path

log = get_logger("live_connect")


class LiveConnect(AbstractContextManager):
    """Connect to a system/resource and enable system activation, deactivation and configuring controller setpoints.

    The class can be initialized directly by supplying all required arguments or from a json configuration file.

    If initializing with the from_json classmethod, the class takes a json config file which defines all of the
    required nodes. The json file should have the following format:

        * **system**: Define the system to be controlled. This is the top level block - it may have multiple dictionary
          members which should each specify all of the following information. If there are multiple systems, all of
          the node names will be prefixed with the system name. For example, if the system name is 'chp' and the node
          name is 'op_request', the full node name will be 'chp.op_request'.
        * **name**: A name to uniquely identify the system
        * **servers**: Servers which are responsible for manageing the system. This section should contain a dictionary
          of servers, identified by a unique name. Each server has the following values:

            * **url**: URL of the server (format: netloc:port, without the scheme)
            * **protocol**: Protocol for the connection, for example opcua
            * **usr**: Username to identify with the server (default: None)
            * **pwd**: Password to login to the server (default: None)
        * **nodes**: The nodes section contains a list of nodes. Each node is specific to one of the servers specified
          above and has a unique name, by which it can be identified. In detail, each node has the following values:

            * **name**: A name to uniquely identify the node.
            * **server**: Identifier of one of the servers defined above
            * **dtype**: Data type of the Node on the server (default: float)
            * Other values must be defined depending on the protocol, for example this could be an **opc_id** or
              corresponding values to identify modbus nodes. More detail about this can be found in connectors.Nodes
        * **set_value**: The set_value is the main value, which should be manipulated. It must have some additional
          information to be processed correctly. This includes:

            * **name**: Name of the node, which is manipulated as the set_value
            * **min**: Minimum value to be allowed
            * **max**: Maximum value to be allowed
            * **threshold**: Activation/Deactivation threshold. The activate action will be executed for values above
              the threshold and the deactivation action will be executed for values below the threshold. (see below)
            * **add**: Scale the set_value by adding a set amount
            * **mult**: Scale the set_value by multiplying by this factor
        * **activation_indicators**: The values specified in this section are used to determine, whether the system is
          currently active. This is a dictionary of nodes and values, these nodes should be compared against. Each node
          is identified by the name specified above. Each node must have the following values:

            * **compare**: Comparison operation to perform (for example '==' or '<=').
            * **value**: Value to compare against. If the result of all comparisons is True, the system is considered
              to be currently active.
        * **observe**: Values to return after each set operation. This is just a list of node names as specified in
          the nodes section above.
        * **actions**: In the actions section, default values for more complex operations are specified. For example,
          if a system needs to be initialized or deactivated. Each of the actions is a dictionary of node names
          and corresponding values, for example {node1: true, node2: 0.2}. The following actions are defined:

            * **init**: Initialize the system. This is used to make sure that the system is ready to receive control
              values from the connector.
            * **activate**: Activate the system. This is used to set the system to an active state, for example by
              requesting the system to start its operation and choosing and operating mode. This can be used to set
              the system up to receive more detailed control values
            * **deactivate**: Execute any actions to deactivate the system
            * **close**: Reset the system. This is used when the connector is closed, to make sure the system is left
              in a safe and clean state.

    If initializing manually, each of the actions and some other information must be specified as parameters to the
    class. This way, the class will not automatically identify different systems and will instead assume that all
    parameters belong to the same system. If system differentiation is required, the node prefixing must be done
    manually. The third option is to initialize from a dictionary using the from_dict function. This works equivalent to
    the from_json function.

    .. warning ::
        Always call the close function after your are done using the connection! This is required even if no nodes
        must be written to reset the system since the connection itself must be closed. Therefore, this class should
        only be called within a try-finally clause. Alternatively the class can be used as a context manager in a with
        statement

    :param name: Name to uniquely identify the system. The name is also used as the default system prefix.
    :param nodes: Sequence/List of Nodes, which should be used for the actions and for initializing connections
    :param init: Nodes for initializing the system (see above, default: None)
    :param activate: Nodes for activating the system (see above, default: None)
    :param deactivate: Nodes for deactivating the system (see above, default: None)
    :param close: Nodes to close the connection and reset the system (see above, default: None)
    :param observe: Nodes to read from and return after each 'set' operation (see above, default: None)
    :param activation_indicators: Nodes that determine, whether the system must be deactivated or activated. This
                                  is used internally in conjunction with the set_value threshold to determin whether
                                  the activate/deactivate methods must be called before setting values.
    :param set_values: Specification of the node which is used for setting the main control value.
    """

    def __init__(
        self,
        nodes: Sequence[Node],
        name: str = None,
        *,
        init: Mapping[str, Any] = None,
        activate: Mapping[str, Any] = None,
        deactivate: Mapping[str, Any] = None,
        close: Mapping[str, Any] = None,
        observe: Sequence[str] = None,
        activation_indicators: Mapping[str, Any] = None,
        set_values: Mapping[str, Any] = None,
    ) -> None:
        #: Name of the system
        self.name: Optional[str] = name.strip() if name is not None else None
        #: Connection objects to the resources
        self._connections: Dict[str, Connection] = connections_from_nodes(nodes)
        #: Mapping of all nodes
        self._nodes: Dict[str, Node] = name_map_from_node_sequence(nodes)
        #: Mapping of node names to connections
        self._connection_map = {node.name: node.url_parsed.hostname for node in self._nodes.values()}

        errors = False
        #: Nodes for initializing the system
        self._init_vals: Optional[Dict[str, Any]] = {}
        if init is not None:
            for key, sys_val in init.items():
                if isinstance(sys_val, Mapping):
                    self._init_vals.update({k: v for k, v in sys_val.items()})
                elif sys_val is not None:
                    self._init_vals[key] = sys_val
        if len(self._init_vals) <= 0:
            self._init_vals = None
        elif not self._init_vals.keys() <= self._nodes.keys():
            log.error("Not all nodes required for initialization are configured as nodes.")
            errors = True

        #: Nodes for closing the connection
        self._close_vals: Optional[Dict[str, Any]] = {}
        if close is not None:
            for key, sys_val in close.items():
                if isinstance(sys_val, Mapping):
                    self._close_vals.update({k: v for k, v in sys_val.items()})
                elif sys_val is not None:
                    self._close_vals[key] = sys_val
        if len(self._close_vals) <= 0:
            self._close_vals = None
        elif not self._close_vals.keys() <= self._nodes.keys():
            log.error("Not all nodes required for closing the object are configured as nodes.")
            errors = True

        #: Nodes to observe
        self._observe_vals: Optional[List[str]] = []
        if observe is not None:
            for sys_val in observe:
                if isinstance(sys_val, Sequence) and type(sys_val) is not str:
                    self._observe_vals.extend(sys_val)
                elif sys_val is not None:
                    self._observe_vals.append(sys_val)
        if len(self._observe_vals) <= 0:
            self._observe_vals = None
        elif not set(self._observe_vals) <= self._nodes.keys():
            log.error("Not all observation nodes of the object are configured as nodes.")
            errors = True

        #: Configuration for the step set value
        self._set_values: Dict[str, Any] = {}
        if set_values is not None:
            nds = set()
            setval = set_values.values() if isinstance(set_values, Mapping) else set_values
            for sys_val in setval:
                self._set_values[sys_val["name"]] = sys_val
                nds.add(self._set_values[sys_val["name"]]["node"])
        if len(self._set_values) <= 0:
            self._set_values = None
        elif not nds <= self._nodes.keys():
            log.error("Not all nodes required for setting control values are configured as nodes.")
            errors = True

        for keys, set_value in self._set_values.items():
            set_value.setdefault("node", set_value["name"])
            set_value.setdefault("threshold", 0)
            set_value.setdefault("mult", 1)
            set_value.setdefault("add", 0)
            set_value.setdefault("min", None)
            set_value.setdefault("max", None)

        #: Nodes for activating the system
        self._activate_vals: Dict[str, Any] = {}
        if activate is not None:
            for key, sys_val in activate.items():
                if sys_val is not None:
                    self._activate_vals[key] = sys_val
                    if not sys_val.keys() <= self._nodes.keys():
                        log.error(
                            f"Not all nodes required for system activation of system {key} are " f"configured as nodes."
                        )
                        errors = True
                else:
                    self._activate_vals[key] = None

        #: Nodes for deactivating the system
        self._deactivate_vals: Dict[str, Any] = {}
        if deactivate is not None:
            for key, sys_val in deactivate.items():
                if sys_val is not None:
                    self._deactivate_vals[key] = sys_val
                    if not sys_val.keys() <= self._nodes.keys():
                        log.error(
                            f"Not all nodes required for system activation of system {key} are " f"configured as nodes."
                        )
                        errors = True
                else:
                    self._deactivate_vals[key] = None

        #: Indicator to keep track of to know whether the system must be activated or deactivated. If this is not
        #: set, the user must keep track of when the activation and deactivation functions need to be called. This
        #: may not be necessary for all systems.
        self._activation_indicators: Dict[str, Any] = {}
        if activation_indicators is not None:
            for key, sys_val in activation_indicators.items():
                if sys_val is not None:
                    self._activation_indicators[key] = sys_val
                    if not sys_val.keys() <= self._nodes.keys():
                        log.error(
                            f"Not all nodes required for system activation of system {key} are " f"configured as nodes."
                        )
                        errors = True
                else:
                    self._activation_indicators[key] = None

        if errors:
            raise KeyError("Not all required nodes are configured.")

        if self._init_vals is not None:
            self.write(self._init_vals)

    @classmethod
    def from_json(cls, files: Union[Path, Sequence[Path]], usr: str = None, pwd: str = None) -> "LiveConnect":
        """Initialize the connection directly from json configuration files. The file should contain parameters
        as described above. A list of file names can be supplied to enable the creation of larger, combined systems.

        Username and password supplied as keyword arguments will take precedence over information given in
        the config file.

        :param file: Configuration file paths. Accepts a single file or a list of files
        :param usr: Username for authenticaiton with the resource
        :param pwd: Password for authentication with the resource
        :return: LiveConnect instance as specified by the json file.
        """
        files = [files] if not isinstance(files, Sequence) else files

        config = {"system": []}
        for file in files:
            file = pathlib.Path(file) if not isinstance(file, pathlib.Path) else file
            result = json_import(file)
            if "system" in result:
                config["system"].extend(result["system"])
            else:
                config["system"].append(result)

        return cls.from_dict(usr=usr, pwd=pwd, **config)

    @classmethod
    def from_dict(cls, usr: str = None, pwd: str = None, **config) -> "LiveConnect":
        """Initialize the connection directly from a config dictionary. The dictionary should contain parameters
        as described above.

        Username and password supplied as keyword arguments will take precedence over information given in
        the config file.

        :param usr: Username for authenticaiton with the resource
        :param pwd: Password for authentication with the resource
        :param config: Configuration dictionary
        :return: LiveConnect instance as specified by the json file.
        """

        _req_settings = {"name": {}, "servers": {"url", "protocol"}, "nodes": {"name", "server"}}

        if "system" not in config or not isinstance(config["system"], Sequence) or len(config["system"]) < 1:
            raise KeyError("Could not find a valid 'system' section in the configuration")

        # Check that all required parameters are present in config
        errors = False
        for system in config["system"]:
            for sect in _req_settings:
                if sect not in system:
                    log.error(f"Required parameter '{sect}' not found in configuration.")
                    errors = True
                else:
                    for name in _req_settings[sect]:
                        sec = system[sect].values() if isinstance(system[sect], Mapping) else system[sect]
                        for i in sec:
                            if name not in i:
                                log.error(f"Required parameter '{name}' not found in config section '{sect}'")
                                errors = True

        if errors:
            log.error("Not all required config parameters were found. Exiting.")
            exit(1)

        # Make sure all of the required sections exist - they are just none if they are not in the file.
        nodes_conf = []
        observe = []
        act_indicators = {}
        set_values = {}
        actions = {"activate": {}, "deactivate": {}, "init": {}, "close": {}}

        for system in config["system"]:
            # Combine config for nodes with server config and add system name to nodes
            for n in system["nodes"]:
                server = system["servers"][n["server"]]

                # Parse the url, make sure the scheme is valid and remove the scheme if present to enable the
                # injection of username and password later on.
                url = urlparse(server["url"])
                scheme = url[0] if url[0] != "" else default_schemes[server["protocol"]]
                url = urlunparse(["", *url[1:6]])

                if usr is not None and pwd is not None:
                    n["url"] = f"{scheme}://{usr}:{pwd}@{url}"
                elif "usr" in server and "pwd" in server:
                    n["url"] = f"{scheme}://{server['usr']}:{server['pwd']}@{url}"
                else:
                    n["url"] = f"{scheme}://{url}"
                n["protocol"] = server["protocol"]
                n["name"] = f"{system['name']}.{n['name']}"
                nodes_conf.append(n)

            # Rename set_value
            set_values[system["name"]] = system["set_value"]
            set_values[system["name"]]["name"] = f"{system['name']}.{system['set_value']['name']}"
            set_values[system["name"]]["node"] = f"{system['name']}.{system['set_value']['node']}"

            # Convert activation_indicators
            if "activation_indicators" in system and system["activation_indicators"] is not None:
                act_i = {}
                for name, value in system["activation_indicators"].items():
                    act_i[f"{system['name']}.{name}"] = value
                act_indicators[system["name"]] = act_i
            else:
                act_indicators[system["name"]] = None

            # Convert observations
            if "observe" in system and system["observe"] is not None:
                observe.extend(f"{system['name']}.{name}" for name in system["observe"])

            # Convert actions
            for action in {"init", "close", "activate", "deactivate"}:
                if "actions" not in system:
                    actions[action][system["name"]] = None
                elif action not in system["actions"] or system["actions"][action] is None:
                    actions[action][system["name"]] = None
                else:
                    act = {}
                    for name, value in system["actions"][action].items():
                        act[f"{system['name']}.{name}"] = value
                    actions[action][system["name"]] = act

        # Initialize the node objects
        nodes = Node.from_dict(nodes_conf)
        name = config["system"][0]["name"] if len(config["system"]) == 1 else None

        return cls(
            nodes,
            name,
            init=actions["init"],
            activate=actions["activate"],
            deactivate=actions["deactivate"],
            close=actions["close"],
            observe=observe,
            activation_indicators=act_indicators,
            set_values=set_values,
        )

    @property
    def nodes(self) -> Mapping[str, Node]:
        """Mapping of all node objects of the connection"""
        return self._nodes.copy()

    def _activated(self, system: str) -> bool:
        """Current activation status of the system, as determined by the activation_indicator

        :param system: System for which activation status should be checked
        """
        check_map = {"==": "__eq__", ">": "__gt__", "<": "__lt__", ">=": "__gte__", "<=": "__lte__"}

        # If activation_indicators are specified, read them and identify the status, otherwise the system is
        # considered to be always active.
        if system not in self._activation_indicators:
            raise KeyError(f"Cannot check status of unknown system {system}")
        elif self._activation_indicators[system] is not None:
            values = self.read(*self._activation_indicators[system].keys())
            results = []
            for name, check in self._activation_indicators[system].items():
                try:
                    results.append(getattr(values[name], check_map[check["compare"]])(check["value"]))
                except KeyError:
                    raise KeyError(f"Unknown comparison operation {check['compare']}")

            activated = sum(results) >= 1
        else:
            activated = True
        return activated

    def step(self, value: Mapping[str, Any]) -> Dict[str, Any]:
        """Take the set_value and determine, whether the system must be activated or deactivated. Then set the value
        and finally read and return all values as specified by the 'observe' parameter.

        :param value: Value to use as the control value/set_value
        :return: Values read from the connection as specified by 'observer' parameter
        """
        write = {}
        for name, val in value.items():
            n = f"{self.name}.{name}" if "." not in name and self.name is not None else name
            if n in self._set_values:
                system, node = n.split(".")

                if (self._set_values[n]["min"] is not None and self._set_values[n]["min"] > val) or (
                    self._set_values[n]["max"] is not None and val > self._set_values[n]["max"]
                ):
                    raise ValueError(
                        f"Set value for node {n} is out of bounds. Value: {val}; "
                        f"Bounds: [{self._set_values[n]['min']}, {self._set_values[n]['max']}]."
                    )

                # Determine activation status and activate or deactivate the system correspondingly
                if val >= self._set_values[n]["threshold"]:
                    self.activate(system)
                elif val <= self._set_values[n]["threshold"]:
                    self.deactivate(system)

                write[self._set_values[n]["node"]] = (val + self._set_values[n]["add"]) * self._set_values[n]["mult"]
            else:
                write[n] = val

        # Write the scaled control value and return the observed values.
        self.write(write)
        return self.read(*self._observe_vals)

    def write(self, nodes: Union[Mapping[str, Any], Sequence[str]], values: Optional[Sequence[Any]] = None) -> None:
        """Write any combination of nodes and values.

        :param nodes: Mapping of Nodes and values or Sequence of node names to write to
        :param values: If nodes are given as a Sequence, this second parameter determined the values to be written.
                       If nodes are given as a Mapping, this parameter is not required. It defaults to None.
        :return: None
        """
        # Determine, whether nodes and values are given as a mapping in the nodes parameter or separately, using both
        # parameters. In the second case, create the mapping.
        if not isinstance(nodes, Mapping) and len(nodes) != len(values):
            raise ValueError(
                f"Each node must have a corresponding value for writing. "
                f"Nodes and values must be of equal length. Given lengths are"
                f"'nodes': {len(nodes)}, 'values': {len(values)}"
            )
        elif not isinstance(nodes, Mapping):
            nodes = dict(zip(nodes, values))

        # Sort nodes to be written by connection
        writes = {url: {} for url in self._connections.keys()}
        for name, value in nodes.items():
            n = f"{self.name}.{name}" if "." not in name and self.name is not None else name
            writes[self._connection_map[n]][self._nodes[n]] = value

        # Write to all selected nodes for each connection
        for connection in self._connections:
            if writes[connection]:
                self._connections[connection].write(writes[connection])

    def read(self, *nodes: str) -> Dict[str, Any]:
        """Take a list of nodes and return their names and most recent values

        :param nodes: One or more nodes to read
        :return: Dictionary of the most current node values.
        """
        # Sort nodes to be read by connection
        reads = {url: [] for url in self._connections.keys()}
        for name in nodes:
            n = f"{self.name}.{name}" if "." not in name and self.name is not None else name
            reads[self._connection_map[n]].append(self._nodes[n])

        # Read from all selected nodes for each connection
        result = {}
        for connection in self._connections:
            if reads[connection]:
                result.update(self._connections[connection].read(reads[connection]).iloc[0].to_dict())

        return result

    def activate(self, system: Optional[str] = None) -> None:
        """Take the list of nodes to activate and set them to the correct values to activate the system

        :param system: System for which should be activated (default: self.name)
        """
        system = self.name if system is None and self.name is not None else system

        if not self._activated(system) and self._activate_vals[system] is not None:
            self.write(self._activate_vals[system])

    def deactivate(self, system: Optional[str] = None) -> None:
        """Take the list of nodes to deactivate and set them to the correct values to deactivate the system

        :param system: System for which should be activated (default: self.name)
        """
        system = self.name if system is None and self.name is not None else system

        if self._activated(system) and self._deactivate_vals[system] is not None:
            self.write(self._deactivate_vals[system])

    def close(self) -> None:
        """Reset the system and close the connection."""
        if self._close_vals is not None:
            self.write(self._close_vals)

    def __enter__(self) -> "LiveConnect":
        return self

    def __exit__(
        self, exc_type: Optional[Type[Exception]], exc_val: Optional[Exception], exc_tb: Optional[types.TracebackType]
    ) -> bool:
        self.close()
        return False
