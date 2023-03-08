.. _install:

Installation
============

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

.. _install_julia:

Using Julia Extensions
-------------------------------------

First, it is necessary to have julia available in your system, the recommended version
is the latest stable version but minimum v.1.8 (`download julia <https://julialang.org/downloads/>`_).

Next, it's important, that you include Julia to the operating system path. For windows 10 do the following steps:

- **1st step**: Open run (Windows Key + R) and write ``rundll32 sysdm.cpl,EditEnvironmentVariables`` .
- **2nd step**: Under either "User Variables" or "System Variables", the row "Path" add with "edit" the Julia path. The Julia
  path should have the following schema: ``C:Users\User\AppData\Local\Programs\Julia-1.8.5\bin``.
- **3rd step**: Click OK and restart eta-utility.

If you are using a different operating system, look here: (`path julia <https://julialang.org/downloads/platform/#windows/>`_)

The next step is to activate your virtual environment for the eta-utility
and inside the eta-utility directory execute the following command:

.. code-block::

    $> install-julia

This command will install PyJulia, configure PyCall in your system, and install the julia extensions package (ju_extensions) from eta_utility.
