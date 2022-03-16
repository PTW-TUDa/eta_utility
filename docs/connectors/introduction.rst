.. _intro_connectors:

Introduction
=================
The *eta_utility.connectors* module is meant to provide a standardized interface for multiple different
protocols which are used in factory operations or for the optimization of factory operations. Two important
protocols which we encounter regularly are **Modbus TCP** and **OPC UA**. In addition to this  we have also created
connectors for additional API which we work with.

The *eta_utility* connector format has the advantage that it can be used for many different kinds of protocols
and APIs, with the limitation that some of them do not support all functions (for example, specific APIs/protocols
may not provide write access). Each connection can contain multiple *Nodes* (see below). These are used as the
default data points when reading data from the connection. Read data will be returned in a
:py:class:`pandas.DataFrame` with the node names as column names.

The *connectors* module also provides subscription handlers which take data read from connections in regular
intervals and for example store it in files or in memory for later access. These subscription handlers can handle
multiple different connections (with different protocols) at the same time.

The *LiveConnect* class is specifically designed to combine the functionality from the *eta_x* and *connectors*
modules. It can establish connections and provides an interface equivalent to the classes in the *simulators*
module. This allows easy substitution of simulation models with actual connections to real machines (or the other
way). When trying to deploy a model into operation this substitution can be very useful.

The connectors are based on the concept of *Nodes* to which we establish connections. A *Node* object is a
unique description of a specific data point and includes all information required to establish a connection and
read data from the specified data point. Each data point has its own node, not just each device (or connection)
that we are connecting to. Therefore, *Nodes* are the easiest way to instantiate connections, however they can
be a bit unwieldy to work with when trying to read many different data points from the same device.


There are multiple ways to instantiate connections, depending on the use case:

- When only a single connection is needed, the connection can be instantiated directly. This is possible
  with or without specifying nodes. *Node* objects do have to be created however, to be able to tell the
  connection where (which data points) to read data from or write data to.
- If you already have the *Node* objects and want to quickly create a single connection, all connections have the
  *from_nodes* classmethod which requires less duplicate information than direct instantiation.
- When a single connection is needed and access to the *Node* objects is not required, many (not all) connectors
  offer a *from_ids* classmethod which returns the Connection object and creates the *Nodes* only internally.
- When multiple connections to different devices are needed, it is usually easiest to create all of the *Node*
  objects first and use the :py:class:`eta_utility.connectors.common.connections_from_nodes` function to
  get an array of connections. The function will automatically select the correct nodes for each connection.

Nodes
----------
Each *Node* object uniquely identifies a specific data point. All *Node* objects have some information in
common. This information idenfies the device which the data point belongs to and can also contain information
required for authentication with the device. Depending on the protocol the *Node* object contains additional
information to correctly identify the data points.

The URL may contain the username and password (``schema://username:password@hostname:port/path``). This is handled
automatically by the connectors and the username and password will be removed before creating a connection.

.. autoclass:: eta_utility.connectors::Node
    :noindex:

The following are there to document the required parameters for each type of node - always use the *Node* class and
specify the protocol to instantiate nodes.

.. autoclass:: eta_utility.connectors.node::NodeLocal
    :inherited-members:
    :exclude-members: get_eneffco_nodes_from_codes, from_dict, from_excel, protocol
    :noindex:

.. autoclass:: eta_utility.connectors.node::NodeModbus
    :inherited-members:
    :exclude-members: get_eneffco_nodes_from_codes, from_dict, from_excel, protocol
    :noindex:

.. autoclass:: eta_utility.connectors.node::NodeOpcUa
    :inherited-members:
    :exclude-members: get_eneffco_nodes_from_codes, from_dict, from_excel, protocol
    :noindex:

.. autoclass:: eta_utility.connectors.node::NodeEnEffCo
    :inherited-members:
    :exclude-members: get_eneffco_nodes_from_codes, from_dict, from_excel, protocol
    :noindex:

.. autoclass:: eta_utility.connectors.node::NodeEntsoE
    :inherited-members:
    :exclude-members: get_eneffco_nodes_from_codes, from_dict, from_excel, protocol
    :noindex:

Connection Instantiation
----------------------------
Connections can be instantiated using different methods as described above. The three most common methods are described
here, they are instantiation of a connection *from_ids*, the instantiation of a connection *from_nodes* and
instantiation of multiple connections using *connections_from_nodes*.

Instantiation from node(s) is useful if you already have created some nodes and would like to create a connection
from them. The following is the from_node function of the base class. Specific implementations may differ regarding
the accepted keyword arguments.

.. autofunction:: eta_utility.connectors.base_classes::BaseConnection.from_node
    :noindex:

The *from_ids* method is helpful if you do not require access to the nodes and just want to quickly create a single
connection.

 .. note::
    This is not available for all connectors, since the concept of IDs does not apply universally. An
    example is shown here. Refer to the API documentation of the connector you would like to use to see if the
    method exists and which parameters are required.

.. autofunction:: eta_utility.connectors::EnEffCoConnection.from_ids
    :noindex:

The *connections_from_nodes* function is useful for the creation of multiple connections at once.

.. autofunction:: eta_utility.connectors::connections_from_nodes
    :noindex:
