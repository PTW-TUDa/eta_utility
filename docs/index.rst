.. utility-functions

Documentation of the ETA Utility-Functions
=============================================
The utility functions are part of the "ETA Industrial Energy Lab" framework. The utilities provide
some easy to use applications for different use cases.

.. note::
  Information for Students:

  Student work stations do not have direct access to machine controllers. If this is needed, please
  discuss options with the research assistants. For example, there are servers accessible via remote
  desktop that do provide access.

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Guide

    /guide/install
    /guide/faq
    /guide/development

Connectors:
-----------
Connectors can be used for connections to servers using different protocols such as OPC UA and Modbus
as well as EnEffCo. Connectors also provide functionality for writing data and for subscription handling.

OPC UA and Modbus TCP/IP connectors for reading out:
   - OPC UA & Modbus nodes from different data sources
   - OPC UA nodes from single datapoint


For recording OPC UA and Modbus data from different PLCs in parallel use subscriptions with a
single handler.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Connectors

   /connectors/connectors
   /connectors/modbus
   /connectors/opcua
   /connectors/eneffco

Servers:
-----------
Server classes can be used to easily create servers. For example the OPC UA Server class enables
the creation of an OPC UA server with simple variable access.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Servers

   /servers/opcua

Simulators:
--------------
Simulators are used to provide utilities for the simulation of functional mockup units.

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Simulators

    /simulators/fmu

Timeseries:
--------------
Functions to load and easily manipulate timeseries data.

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Timeseries

    /timeseries/timeseries

ETA-X:
---------------
ETA-X provides an optimization framework, based on the OpenAI gym model.

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: ETA-X

    /eta_x/eta_x
    /eta_x/envs
    /eta_x/agents
    /eta_x/mpc_basic
    /eta_x/nsga2
    /eta_x/rule_based

Citing ETA utility-functions
----------------------------
To cite this project in publications:

.. code-block:: bibtex

    @misc{misc,
      author = {ETA-Fabrik},
      title = {ETA Utility functions},
      year = {2020},
      publisher = {GitLab},
      howpublished = {\url{www.gitlab.com}},
    }

Contributing
------------
If you are interested to contribute any improvements or new functions, please contact
the repositories maintainers. See README.md for further instructions!


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
