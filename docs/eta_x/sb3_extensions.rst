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

Extractors
--------------
Extractors are based on :py:class:`stable_baselines3.common.base_class.BaseFeaturesExtractor`.

.. autoclass:: eta_utility.eta_x.common::MLPCNNExtractor
    :noindex:

    The architecture of the feature extractor is controlled by the ``extractor_arch`` parameter.
    It is able to handle observations which consists of classic, time-independent data **and** multiple time
    series. The network is shared between the action and value network.

     .. warning::
        The user must ensure the correct order of observations. The observations used for the cnn part must be located at
        the end of the observation space. The different time-series in the cnn observations must be one by one:
        ((mlp-observations), ((time-series 1), (time-series 2), ...))
        See the example for a more detailed explanation.

    .. figure:: figures/Example_Advanced_extractors.png
        :alt: Example and Concept of the Advanced Extractors

    This graphic shows a possible architecture, which can be built with this class.
    The corresponding net arch parameter for the combined part is:
    net_arch = [500, dict(pi=[400, 300], vf=[400, 300])]
    The net arch for the advanced extractor is:
    extractor_arch = [dict(cnn=[[18, 3],[3, 32, 1, 'valid'],[2, 16, 1, 'valid'],[6, 16, 2, 'valid']], mlp=[100])].
    You can also use the *MLPCNNNetArch* class to determine the architecture.
    The blue boxes mark the 3 parts of the neural network: CNN, MLP and combined. Red circles and boxes show
    parameters, which can be defined by the user. For a simple visualization, only one forward pass is displayed.
    18 of the total 27 observations are used for the CNN part.

     .. note::
        In this example, these 18 observations are 3 different
        time series, which are located one after the other in the observation vector:
        [p0_(15 min), p0_(30 min), p0_(45 min),..., p3_(345 min), p3_(360 min)]
        with p0,p1,p2 as different time series ranging von 15 min - 360 min

    The observations are reshaped based on the user input 3 (number time series). For the CNN part,
    2 filters are specified. The first one has a width of 3 and 32 output channels, which are determined
    by the number of filters. The stride is 1
    and the padding is ‘valid’, so that the respective dimension is reduced from 6 to 4.
    The second filter has a width of 2 and 16 output channels. The height of the filter is determined by the number of
    channels of the input. With a stride of 1 and ‘valid’-padding, the output is a tensor with width 3 and 16 output
    channels.
    Note that the filters are shown in red, and the white tensors represent the shape of the output or input. The actual
    dimension and shape of the tensors may vary. The graphic is intended as a simplified visualization of the different
    layers.

     .. note::
        A pooling layer can be specified by a dict entry as follow: [pooling type, pooling size, stride]. The pooling types
        ‘max_pooling’ and ‘average_pooling’ are implemented.

    The MLP part consists of one layer with 100 neurons. The input of the layer are the 9 observations at the beginning
    of the observation vector, which are determined the total numer of observations (27) and the number of observations,
    used for the CNN part (18). It is also possible to define any number of hidden layers or none. The entry in the
    dictionary is optional.
    The output of the CNN part is flattened and concatenated with the output of the MLP part and will then be the
    input of the combined part, which is specified as usual by the net_arch parameter. Here, the hidden layers can be
    shared between the policy and value network, or by specifying a dict, it can be divided, as it is in this example:
    The shared layer has 500 neurons, and the output is then divided. The policy and value network both have 2 hidden
    levels with 400 and 300 neurons.

.. autoclass:: eta_utility.eta_x.common::MLPCNNNetArch
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
