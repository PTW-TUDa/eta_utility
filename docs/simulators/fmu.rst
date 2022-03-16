.. _fmu_simulators:

Functional Mockup Units
====================================================
Functional Mockup Units (FMUs) are packaged simulation models. The standard makes it quite easy to use simulation
models and it can be exported by many different simulation software packages. More information about the
standard can be found on the `FMI Homepage <https://fmi-standard.org/>`_. The *simulators* in *eta_utility* are
based on the `fmpy package <https://fmpy.readthedocs.io/en/latest/>`_.

*fmpy* is a python implementation of the standard and offers limited additional functionality. For this reason and
because the standard is written for C, fmpy is neither very pythonic nor very easy to use. Therefore, *eta_utility*
implements wrappers which allow for the simulation of both ModelExchange and Co-Simulation FMU models. The wrappers
have an interface similar to *LiveConnect* with regard to the *step* function. This enables easy substitution of
simulation models with direct connections to actual devices.

.. autoclass:: eta_utility.simulators::FMUSimulator
    :members:
    :noindex:

The *FMU2MESlave* is a wrapper for ModelExchange FMUs which provides similar functionality to Co-Simulation FMUs
while using the Cvode solver from *fmpy*.

.. autoclass:: eta_utility.simulators::FMU2MESlave
    :members:
    :noindex:
