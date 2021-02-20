[![pipeline status](https://git.rwth-aachen.de/eta-fabrik/industrialenergylab/utility-functions/badges/master/pipeline.svg)](https://git.rwth-aachen.de/eta-fabrik/industrialenergylab/utility-functions/-/commits/master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)


# ETA Utility-Functions

The utility functions are part of the "ETA Industrial Energy Lab" framework. The utilities provide
some easy to use applications for different use cases.

## Connectors:

Connectors can be used for connections to servers using different protocols such as OPC UA and Modbus
as well as EnEffCo. Connectors also provide functionality for writing data and for subscription handling.

OPC UA and Modbus TCP/IP connectors for reading out:

   - OPC UA & Modbus nodes from different data sources
   - OPC UA nodes from single datapoint

For recording OPC UA and Modbus data from different PLCs in parallel use subscriptions with a
single handler.

## Servers

Server classes can be used to easily create servers. For example the OPC UA Server class enables
the creation of an OPC UA server with simple variable access.

## Simulators

Simulators are used to provide utilities for the simulation of functional mockup units.

## Citing ETA utility-functions

To cite this project in publications:

    @misc{misc,
      author = {ETA-Fabrik},
      title = {ETA Utility functions},
      year = {2020},
      publisher = {GitLab},
      howpublished = {\url{www.gitlab.com}},
    }

## Contributing

If you are interested to contribute any improvements or new functions, please contact
the repositories maintainers:
    Benedikt Grosch
    Martin Lindner
