.. _wetterdienst_connection:

Wetterdienst Connection
====================================================
WetterdienstPrediction
----------------------------------------------------
.. autoclass:: eta_utility.connectors::WetterdienstPredictionConnection
    :members:
    :noindex:

NodeWetterdienstPrediction
----------------------------------------------------
.. autoclass:: eta_utility.connectors.node::NodeWetterdienstPrediction
    :inherited-members:
    :exclude-members: get_eneffco_nodes_from_codes, from_dict, from_excel, protocol, as_dict, as_tuple, evolve
    :noindex:

WetterdienstObservation
----------------------------------------------------
.. autoclass:: eta_utility.connectors::WetterdienstObservationConnection
    :members:
    :noindex:

NodeWetterdienstObservation
----------------------------------------------------
.. autoclass:: eta_utility.connectors.node::NodeWetterdienstObservation
    :inherited-members:
    :exclude-members: get_eneffco_nodes_from_codes, from_dict, from_excel, protocol, as_dict, as_tuple, evolve
    :noindex:

Example
----------------------------------------------------
An example using the **Wetterdienst connection**:

.. literalinclude:: ../../examples/connectors/read_series_wetterdienst.py
    :start-after: --begin_wetterdienst_doc_example--
    :end-before: --end_wetterdienst_doc_example--
    :dedent:
