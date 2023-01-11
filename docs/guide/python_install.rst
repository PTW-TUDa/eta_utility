.. _python_install:

Set up Python and Git
======================

This section explains how to set up Python and Git and also how to create a virtual environment. Additional information can be found in the `Beginner's Guide to Python (external link) <https://wiki.python.org/moin/BeginnersGuide>`_.
If you have already set this up, you can go to the :ref:`_install` section.

Installation of Python
------------------------

If you do not already have it, install Python (64 Bit) on your device. We recommend a version between
3.7 and 3.9. Depending on the used modules newer versions will also work. If you are using the
*eta_x* module, the version is limited by what is supported by *pytorch*.

Below, the installation is illustrated using Python 3.7, 64 Bit. You can install Python for
Windows from `python.org <https://www.python.org/downloads/windows/>`_

.. figure:: figures/1_DownloadFiles.png
   :scale: 35 %
   :alt: Picture of possible download files from python.org.

   Selection of download files from python.org.

When installing, select the "Add Python to PATH" option and also make sure that "pip" is installed.
In addition to this, it typically makes sense to "Add Python to environment variables" which allows
other programs to find it. The setup process and selected options are shown in the following
figures.

.. |bild1| image:: figures/2_Install.png
   :width: 330
   :alt: install
.. |bild2| image:: figures/3_Pip.png
   :width: 330
   :alt: pip
.. |bild3| image:: figures/4_PATH.png
   :width: 330
   :alt: PATH
.. |bild4| image:: figures/5_Finish.png
   :width: 330
   :alt: finish


|bild1| |bild2|
|bild3| |bild4|


.. note::
    Using **Anaconda** is not supported and can lead to weird errors. Especially Anaconda should
    not be combined with a version of CPython on the same machine.


.. note::
    It is recommended to install to a **virtual environment** so that the installed package
    versions do not conflict with other python projects on the computer.

.. _install_git:

Installation of Git
----------------------------------------------

If you do not already have it, install Git on your device. For Windows, you can download
it from `git-scm.com <https://git-scm.com/download/win>`_. Alternatively, it is also
possible to use GUIs (Graphical User Interfaces) for git such as SourceTree or `GitHub Desktop <https://desktop.github.com/>`_. If you have problems with git, see the :ref:`faq`.

.. _create_virtual_environment:

Creating a virtual environment (recommended)
-----------------------------------------------------

A **virtual environment** is a Python environment which isolates the Python interpreter, libraries,
and scripts installed in it from those installed in other virtual environments, and (by default)
from any libraries installed in the "system" Python (The main Python environment installed on your
operating system).

In order to use a virtual environment, it must first be created in an arbitrary directory
on the computer. Open a terminal (for example cmd or PowerShell on Windows) and execute the
following commands.

.. code-block:: console

    $> Python -m venv <Any folder>/venv

Navigate to your chosen directory, then activate the virtual environment:

.. code-block:: console

    $> venv/scripts/activate

The creation and activation of the environment are shown in the following figure.

.. figure:: figures/6_ActivateVE.png
   :width: 700
   :alt: Activate virtual environment

   Create and activate virtual environment.

When the virtual environment is activated, *(venv)* is prefixed to the console line.
The commands in the following chapters can be executed in the virtual environment without
any adjustments.

.. note::

    Some IDEs (Integrated Development Environments) such as PyCharm will automate the following process
    for you.
