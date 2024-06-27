.. _emonio_connection:

EmonioConnection
======================================
Eta-utility provides a connection class to read data from an Emonio device.
The emonio has to be in the same network as the computer running eta-utility.
Since the EmonioConnection is implmented using the modbus protocol, the Emonio needs
to have its Modbus Server configured to `Enabled`.

This can be done in the Emonio App under `Settings` -> `Modbus Server`.
The ip address and port of the emonio can also easily be found in the app.

See the Emonio documentation for more information:
https://wiki.emonio.de/de/Emonio_P3

.. autoclass:: eta_utility.connectors::EmonioConnection
    :members:
    :noindex:

.. _emonio_available_nodes:

.. list-table:: Available Emonio Nodes
    :widths: 10 90
    :header-rows: 1

    * - Allowed Names
      - Address
    * - 'VRMS', 'V_RMS', 'Voltage', 'V', 'Spannung'
      - 0
    * - 'IRMS', 'I_RMS', 'Current', 'I', 'Strom'
      - 2
    * - 'WATT', 'Power', 'W', 'Leistung', 'Wirkleistung'
      - 4
    * - 'VAR', 'Reactive Power', 'VAR', 'Blindleistung'
      - 6
    * - 'VA', 'Apparent Power', 'VA', 'Scheinleistung'
      - 8
    * - 'FREQ', 'Frequency', 'Hz', 'Frequenz'
      - 10
    * - 'KWH', 'Energy', 'kWh', 'Energie'
      - 12
    * - 'PF', 'Power Factor', 'PF', 'Leistungsfaktor'
      - 14
    * - 'VRMS MIN', 'VRMS_MIN', 'Voltage Min', 'V Min', 'Spannung Min'
      - 20
    * - 'VRMS MAX', 'VRMS_MAX', 'Voltage Max', 'V Max', 'Spannung Max'
      - 22
    * - 'IRMS MIN', 'IRMS_MIN', 'Current Min', 'I Min', 'Strom Min'
      - 24
    * - 'IRMS MAX', 'IRMS_MAX', 'Current Max', 'I Max', 'Strom Max'
      - 26
    * - 'WATT MIN', 'WATT_MIN', 'Power Min', 'W Min', 'Leistung Min'
      - 28
    * - 'WATT MAX', 'WATT_MAX', 'Power Max', 'W Max', 'Leistung Max'
      - 30
    * - 'Temp', 'degree', 'Temperature', '°C', 'Temperatur'
      - 500
    * - 'Impulse', 'Impuls'
      - 800


.. note::
    For reading MIN and MAX values, the phase must be specified. "abc" is not allowed.

.. autoclass:: eta_utility.connectors.emonio::NodeModbusFactory
    :members:
    :noindex:


Examples
--------------------------------------
This example demonstrates how to create a :class:`~eta_utility.connectors.live_connect.LiveConnect` from a dictionary to read data from an Emonio device.
Alternatively, the LiveConnection can be created from a JSON file, with the JSON having the same structure as the dictionary.

``url`` has to be replaced with the IP address and port of the Emonio device, e.g. ``"192.168.178.123:502"``.

.. literalinclude:: ../../examples/connectors/read_emonio_live.py
    :start-after: --live--
    :end-before: --live--
    :dedent:

Here we create the Emonio nodes manually and read them with the
:class:`~eta_utility.connectors.emonio.EmonioConnection` class.

.. literalinclude:: ../../examples/connectors/read_emonio_live.py
    :start-after: --emonio--
    :end-before: --emonio--
    :dedent:

The :class:`~eta_utility.connectors.emonio.NodeModbusFactory` can be used to create modbus nodes directly.
But this is not recommended, as the :class:`~eta_utility.connectors.emonio.EmonioConnection` class is more convenient and has extra error checking.

.. literalinclude:: ../../examples/connectors/read_emonio_live.py
    :start-after: --modbus--
    :end-before: --modbus--
    :dedent:
