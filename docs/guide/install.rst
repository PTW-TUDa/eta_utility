.. _install:

Installation
============

If you need to install python and create a virtual environment first, please take a look at the
python installation guide:

- :ref:`python_install`
- `Beginner's Guide to Python (external link) <https://wiki.python.org/moin/BeginnersGuide>`_


This section explains how to install the ETA Utility Functions for usage only. For instructions
what to consider during the installation process if you want to contribute to development of
the utility functions, please see the development guide :ref:`development`.

You can install the basic package (without *eta_x*) or the entire library, both options are
shown below.


The installation is performed using pip:

.. code-block:: console

   $> pip install eta_utility

There are multiple classes of optional requirements. If you would like to use some of the optional components, please install one or more of the following:

- *eta_x*: Contains dependencies for the optimization part of the framework
- *examples*: Dependencies required to run the examples
- *develop*: All of the above and additional dependencies for the continuous integration processes. Required when performing development work on eta_utility.

The optional requirements can be installed using pip. For example:

.. code-block:: console

   $> pip install eta_utility[eta_x]
