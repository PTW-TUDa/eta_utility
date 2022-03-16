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

If you need to use *eta_x* agents or environments, specify install option "eta_x".

.. code-block:: console

   $> pip install .[eta_x]
