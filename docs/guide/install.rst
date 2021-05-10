.. _install:

Installation
============

Installation of Python
######################

If you do not already have it, install Python (64 Bit) on your device. We recommend a version between 3.6.8 and 3.8.10. Newer versions should also work; however, they have not been tested in this context yet.
The installation is below illustrated choosing Python 3.7, 64 Bit. You can install Python for Windows `here <https://www.python.org/downloads/windows/>`_ (e.g., an installation of Python 3.7.9).



.. figure:: Pictures\\Bild1.png
   :scale: 35 %
   :alt: Picture of possible download files

   Possible download files to choose


Do not forget to add Python to the environment variables of the device (in other words: Add Python to PATH). You should also install pip.

.. |bild1| image:: Pictures\\Bild4.png
   :width: 330
   :alt: install
.. |bild2| image:: Pictures\\Bild2.png
   :width: 330
   :alt: pip
.. |bild3| image:: Pictures\\Bild3.png
   :width: 330
   :alt: PATH
.. |bild4| image:: Pictures\\Bild5.png
   :width: 330
   :alt: finish


|bild1| |bild2|
|bild3| |bild4|

.. note::
    This guide does not fully apply when you are using Anaconda. In such a case, Anaconda must not be combined with a normal Python version. In addition, a workaround may have to be found for many following executions, such as the *pre-commit install*.

(Optional) Installation of Microsoft MPI and Visual Studio Build Tools
######################################################################

The requirement of installing MPI should now have been removed with the usage of a newer stable baseline version (v.3).
However, for some reasons and troubleshooting you may need to install Microsoft MPI and/or VS Build Tools (last one only for ETA-X extension).

Please **skip this step** at first and let us know if it worked despite that. If not, an installation of one of the following tools might help:


* `Visual Studio Build Tools <https://visualstudio.microsoft.com/de/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16>`_ (incl. C++ build tools)

    .. figure:: Pictures\\Bild6.png
       :scale: 30 %
       :alt: What to choose during installation

* `MPI <https://www.microsoft.com/en-us/download/details.aspx?id=57467>`_ - ATTENTION: install the msmpisetup.exe, not the .msi-file!


Installation of Project ETA Utility Function
############################################

This section explains how to install the ETA Utility Functions. You can install Utility Functions for **usage-only** or as a **developer**.
In addition to the basic tool, it is also possible to install two different extensions: the **ETA-X extension** and the **development extension** (only for developer option).




Usage-Only option
*****************


//TODO: Update and check if that is right

Installation with pip out of git repository:

.. code-block:: console

   pip install eta_utility@ git+https://.....

Or local file:
Open cmd and navigate to the directory where the utility-functions folder is located.
The package can be installed via pip using:

.. code-block:: console

   pip install .

If you need to use ETA-X agents or environments, specify install option “eta_x”.

.. code-block:: console

   pip install .[eta_x]




Developer option
****************

As a developer, you first have to create a connection to the respective GitLab project before installing utility functions.



Installation of Git and cloning the repository
----------------------------------------------

If you do not already have it, install Git on your device. For Windows, you can download it from `here <https://git-scm.com/download/win>`_. Alternatively, it is also possible to use Git programs such as GitHub Windows or `GitHub Desktop <https://desktop.github.com/>`_.

If you already use GitHub Windows or `GitHub Desktop <https://desktop.github.com/>`_, you already have Git on your device, but Git will not necessarily be assigned to PATH.
In this case, you can skip the first Git-download from above and just add the path of git.exe to your environment variables by yourself (here is shown the path for git.exe in GitHub Desktop 2.6.3 added to the SYSTEM variables):


.. figure:: Pictures\\Bild7.png
   :scale: 11 %
   :alt: Adding git to PATH

   Adding git to PATH

In case of any problems with the location of git.exe on your device, `this <https://stackoverflow.com/questions/11928561/where-is-git-exe-located>`_ discussion might help.
After that, clone the repository of the git project on your device.



Installation of ETA Utility Functions
-------------------------------------

For the next steps, open cmd.

.. warning::
    Depending on where the relevant folders for the installation are located on your OS, cmd may need to be run as administrator.

.. note::
    Optionally, you can create a virtual environment and work inside that. To create one, type:

    .. code-block:: console

       Python -m venv .venv
       Venv\scripts\activate

In case you haven't updated for a while, you might update the most important parts (pip and setuptools) beforehand. Write these commands in cmd:

.. code-block:: console

   Python -m pip install --upgrade pip
   pip install --upgrade setuptools


After this, go to the root directory of the Git project and install the project using:

.. code-block:: console

   pip install -e .

.. image:: Pictures\\Bild8.png
   :width: 700
   :alt: cmd install

It might be that you will be asked again for your Git-Login.

.. image:: Pictures\\Bild9.png
   :width: 300
   :alt: git login

This installation can take a while.



(Optional) Installation of Extensions
-------------------------------------

Now you can also install an extension.


*➤Development Extension*
^^^^^^^^^^^^^^^^^^^^^^^^^

For a developer, it is recommended to install this extension.

For installing the development extension, write:

.. code-block:: console

   pip install -e .[develop]

in the command line.

Finally, do not forget to execute the following:

.. code-block:: console

   pre-commit install

before performing the first commits to the repository. This ensures that all pre-commit scripts will run correctly.

.. image:: Pictures\\Bild10.png
    :width: 500
    :alt: git not added to PATH


.. note::

   If this comes up instead, you probably did not properly mapped the PATH of git in your system variables (see :ref:`Installation of Git and cloning the repository`):

   .. image:: Pictures\\Bild11.png
       :width: 700
       :alt: git not added to PATH


*➤ETA-X Extension*
^^^^^^^^^^^^^^^^^^^

If you need to use ETA-X agents or environments, specify the installation as follows.

For installing the ETA-X extension, write:

.. code-block:: console

   pip install -e .[eta_x]

in the command line.



Epilogue
--------

To edit the code efficiently, install an IDE for Python (e.g., PyCharm, Visual Studio Code or Visual Studio Community).
If you want to know if the installation was successful, you can run the tests in the *utility-functions* ➔ *test* folder. (These tests also run automatically on each git commit/push.)
