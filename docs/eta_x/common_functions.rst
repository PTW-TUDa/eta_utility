.. _eta_x_common:

Common Functions
==================
In addition to the *ETAx* class, which completely automates the rolling horizon optimization process and the learning
process for reinforcement learning algorithms, the *eta_x* module also provides functions which simplify the creation
of optimization runs and the required environments and algorithms. These functions also provide the interfaces for
reading *eta_x* configuration files and for the logging capabilities of *eta_x*.

Instantiating Environments
--------------------------------
Environments can be instantiated with the *vectorize_environment* function. The function will automatically wrap the
environments with normalization wrappers and monitoring wrappers if required and it can create both interaction
environments and normal environments.

.. autofunction:: eta_utility.eta_x.common::vectorize_environment
    :noindex:

The *vectorize_environment* function will automatically add a callback function as a parameter during
environment instantiation. The callback should be called by the environment after each episode and
after each step. It will create logging output depending on the configuration.

The callback generally used by *ETAx* is *CallbackEnvironment*, which allows for logging in specified
intervals. A *callback* can be anything that is callable and takes an environment instance as its only argument.

.. autoclass:: eta_utility.eta_x.common::CallbackEnvironment
    :noindex:
    :members: __call__

Instantiating Algorithms
------------------------------
Algorithms / models or agents can be instantiated with the *initialize_model* and *load_model* functions.
The *initialize_model* function creates a new model from scratch, while the *load_model* function will load an
existing model from a file created by a *stable_baselines3* algorithm. Both functions will ensure that parameters
passed to the algorithm and that logging output is correctly initialized.

.. autofunction:: eta_utility.eta_x.common::initialize_model
    :noindex:

.. autofunction:: eta_utility.eta_x.common::load_model
    :noindex:

Logging information
---------------------
There are also functions for logging information about the optimization runs, such as the configuration and
the network architecture.

.. autofunction:: eta_utility.eta_x.common::log_run_info
    :noindex:

.. autofunction:: eta_utility.eta_x.common::log_net_arch
    :noindex:

Other helpful functions
-----------------------------
.. automodule:: eta_utility.eta_x.common
    :members:
    :exclude-members: vectorize_environment, CallbackEnvironment, initialize_model, load_model, log_run_info, log_net_arch
    :noindex:
