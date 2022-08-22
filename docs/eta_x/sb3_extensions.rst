.. _sb3_extensions:

Extensions for *stable_baselines3*
===================================
*eta_x* implements some extensions for *stable_baselines3* such as additional feature extractors, policies
and schedules. More information about the prior two can be found in the *stable_baselines3* documentation for
`custom policy networks. <https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html>`_.

In short, *stable_baselines3* divides the policy network into two main parts:

    * A feature extractor which can handle different types of inputs (apart from images).
    * A (fully-connected) network that maps the features to actions and values.
      (controlled by the net_arch parameter)

Policies
----------------------
Some of the agents defined in *eta_x* do not require the specification of a policy. For this special case
you can use the *NoPolicy* class which just does nothing... NoPolicy inherits from
:py:class:`stable_baselines3.common.policies.BasePolicy`.

.. autoclass:: eta_utility.eta_x.common::NoPolicy
    :noindex:

Schedules
-----------------
Schedules evolve over time throughout the learning process of RL applications. *eta_x* implements a BaseSchedule
class which enables the creation of new schedules by inheriting from it and implementing a custom *value* function
which returns the output value, base on an input between 0 and 1. See `learning rate schedules
<https://stable-baselines3.readthedocs.io/en/master/guide/examples.html?highlight=schedule#learning-rate-schedule>`_
in *stable_baselines3*.

The Schedule object is callable so you can pass it directly as a schedule function.

The linear schedule implements a linear evolution of the learning rate.

.. autoclass:: eta_utility.eta_x.common::LinearSchedule
    :noindex:

    Usage:

    .. code-block::

        schedule = LinearSchedule(0.9, 0.2)
        schedule(0.5) == 0.55  # True

Extractors
--------------
Extractors are based on :py:class:`stable_baselines3.common.base_class.BaseFeaturesExtractor`. Use of a custom extractor is specified as a configuration option of the *Policy*. It is specified in the *agent_specific* section of the configuration as part of the *policy_kwargs* dictionary. The required parameters are *features_extractor_class* which must contain the python class and *features_extractor_kwargs* which is a Mapping of the arguments passed to the feature extractor.

.. autoclass:: eta_utility.eta_x.common::CustomExtractor
    :noindex:

    The architecture of the feature extractor is controlled by the ``net_arch`` parameter.
    It is able to handle observations which consist of classic, time-independent data **and** multiple time series. See below for an explanation of how this parameter is interpreted using :py:func:`eta_utility.eta_x.common.common.deserialize_net_arch`.

     .. warning::
        The user must ensure the correct order of observations since the network architecture is often highly dependent on the type of observations passed into the extractor.

Configuring custom neural network architectures
------------------------------------------------
The following function can be used to configure custom neural network architectures in the configuration used by eta_x. This function is used by custom extractors to interpret the ``net_arch`` parameter.

.. autofunction:: eta_utility.eta_x.common::deserialize_net_arch
    :noindex:

Data Processors
------------------
The network architectures used for the CustomExtractor can be extended with the data processors provided py the processors module.

.. autoclass:: eta_utility.eta_x.common::Split1d
    :noindex:

.. autoclass:: eta_utility.eta_x.common::Fold1d
    :noindex:
