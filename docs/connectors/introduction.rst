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

.. _connection instantiation:

There are multiple ways to instantiate connections, depending on the use case:

- When only a single connection is needed, the connection can be instantiated directly. This is possible
  with or without specifying nodes. *Node* objects do have to be created however, to be able to tell the
  connection where (which data points) to read data from or write data to.
- If you already have the *Node* object(s) and want to create connection(s), you should use the *from_node* method of
  the *BaseConnection* class. It requires less duplicate information than direct instantiation. If a single node is
  passed, the method returns the connection for that node. When multiple connections to different devices are needed, it
  is usually easiest to create all of the *Node* objects first and pass them in a list. Then, *from_node* returns a
  dictionary of connections and automatically selects the correct nodes for each connection.
- When a single connection is needed and access to the *Node* objects is not required, many (not all) connectors
  offer a *from_ids* classmethod which returns the Connection object and creates the *Nodes* only internally.

Nodes
----------
Each *Node* object uniquely identifies a specific data point. All *Node* objects have some information in
common. This information idenfies the device which the data point belongs to and can also contain information
required for authentication with the device. Depending on the protocol the *Node* object contains additional
information to correctly identify the data points.

The URL may contain the username and password (``schema://username:password@hostname:port/path``). This is handled
automatically by the connectors and the username and password will be removed before creating a connection.

The *Node* class should always be used to instantiate nodes. The type of the node can be specified using the
*protocol* parameter.

.. autoclass:: eta_utility.connectors::Node
    :noindex:

The following classes are there to document the required parameters for each type of node.

 .. note::
     Always use the *Node* class to instantiate nodes! (not its subclasses)

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
Connections can be instantiated using different methods as described :ref:`above <connection instantiation>`. The two most common methods are described
here, they are instantiation of a connection *from_ids* and the instantiation of a single or multiple connection(s)
using *from_node*.

Instantiation *from_node* in BaseConnection is useful if you already have created some node(s) and would like to create connection(s)
from them. Each connection class also has its own *_from_node* method, since the necessary/accepted keywords might differ. To create connections, a password
and a username are often required. For setting these the following prioritization applies:

- If a password is given in the node, take it.
- If there is no password there, take as "default" a password from the arguments.
- If there is neither, the username and password are empty.

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
