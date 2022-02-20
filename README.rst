ETA Utility Functions
======================

While there are many tools which are useful in the area of energy optimized factory operations, at the
`ETA-Fabrik <https://eta-fabrik.de>`_ at Technical University of Darmstadt we have recognized a lack of
comprehensive frameworks which combine functionality for optimization, simulation and communication with
devices in the factory.

Therefore, we developed the *eta_utility* framework, which provides a standardized interface for the development
of digital twins of factories or machines in a factory. The framework is based on the OpenAI gym environment
and follows a rolling horizons optimization approach. It provides standardized connectors for multiple
communication protocols including OPC UA and Modbus TCP. These facilities can be utilized to easily implement
rolling horizons optimizations for factory systems and to directly control devices in the factory with the
optimization results.

Full Documentation can be found on the
`Documentation Page <https://eta-fabrik.pages.rwth-aachen.de/industrialenergylab/utility-functions/>`_.

The package *eta_utility* consists of five main modules and some addtional functionality:

- *eta_x* is the rolling horizons optimization module which combines the functionality of the
  other modules. It is based on the OpenAI *gym* framework and utilizes algorithms and functions
  from the *stable_baselines3* package. *eta_x* also contains extended base classes for
  environments and additional agents (or algorithms).
- The *connectors* module provides a standardized way to connect to machines and devices in a
  factory or other factory systems (such as energy management systems). The **connectors** can also
  handle subscriptions, for example to regularly store values in a database.
- The *servers* module can be used to easily instantiate servers, for example to publish optimization
  results.
- *simulators* are interfaces based on the *fmpy* package which provide a way to simulate FMU
  (Functional Mockup Unit) models.
  The  *simulators* can be used to perform quick complete simulations or to step through simulation
  models, as would be the case in rolling horizons optimization.
- *timeseries* is an interface based on the *pandas* package to load and manipulate timeseries data
  from CSV files. It can for example rename columns, resample data in more complex ways such as
  multiple different resampling intervals or select random time slices from data. The *scenario_from_csv* function combines much of this functionality.
- Other functionality includes some general utilities which are available on the top level of the
  package.

Some particularities
----------------------

If you want to have logging output from eta utility, call:

.. code-block::

    from eta_utility import get_logger
    get_logger()

**eta_utility** uses dataframes to pass timeseries data and the dataframes are ensured to
contain timezone information where sensible.

Citing this project
--------------------

Please cite this project using our publication:

.. highlight:: none
.. code-block::

    Grosch, B., Ranzau, H., Dietrich, B., Kohne, T., Sossenheimer, J., Lindner, M.,
    Fuhrländer-Völker, D., Weigold, M., Framework for researching energy optimization of factory
    operations. Energy Inform ... Publication in review.

We would like to thank the many contributors who developed functionality for the package, helped with
documentation or provided insights which helped to create the framework architecture.

- *Niklas Panten* for the first implementation of the rolling horizons optimization now available in
  eta_x,
- *Nina Strobel* for the first implementation of the connectors,
- *Thomas Weber* for contributions to the rolling horizons optimization with MPC algorithms,
- *Guilherme Fernandes*, *Tobias Koch*, *Tobias Lademann*, *Saahil Nayyer*, *Magdalena Patyna*, *Jerome Stock*,
- and all others who made small and large contributions.

Contributions
--------------------

If you would like to contribute, please create an issue in the repository to discuss suggested
once the general idea has been agreed upon, you can create a merge request from the issue and
implement your changes there.