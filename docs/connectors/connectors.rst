.. _connectors:

Connections
====================================================
Multiple protocols and APIs are supported by *eta_utility*. Most functionality is common to all
connectors, some of the connectors also offer specific functionality, which only applies to the protocol or
API they implement.

A simple example using the **EnEffCo connection**:

.. literalinclude:: ../../examples/connectors/read_series_eneffco.py
    :start-after: --main--
    :end-before: --main--
    :dedent:

.. autoclass:: eta_utility.connectors::OpcUaConnection
    :members:
    :noindex:

.. autoclass:: eta_utility.connectors::ModbusConnection
    :members:
    :noindex:

.. autoclass:: eta_utility.connectors::EnEffCoConnection
    :members:
    :noindex:

An example using the **ENTSO-E connection**:

.. literalinclude:: ../../examples/connectors/read_series_entsoe.py
    :start-after: --begin_entsoe_doc_example--
    :end-before: --end_entsoe_doc_example--
    :dedent:

.. autoclass:: eta_utility.connectors::ENTSOEConnection
    :members:
    :noindex:
