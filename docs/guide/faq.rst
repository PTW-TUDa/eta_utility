.. _faq:

Frequently Asked Questions
==============================

This is a collection of some questions that have been asked frequently.

I installed git, but the git executable cannot be found
--------------------------------------------------------------
Sometimes, after you install git it will not be automatically assigned to the environment variables.
In this case, you can add the path to git to your environment variables manually. The error might look like the following.

.. figure:: figures/12_ErrorGitNotAddedToPath.jpg
    :width: 700
    :alt: git not added to PATH

    Error message if git was not correctly added to the environment PATH.

On Windows this is done by navigating to "Advanced System Settings", then "Advanced" and clicking
"Environment Variables" at the bottom of the window. In the figure you can see the path for
git.exe in GitHub Desktop 2.6.3 added to the SYSTEM variables:


.. figure:: figures/8_AddGitToPath.png
   :width: 700
   :alt: Adding git to PATH

   Adding git to environment variables.

Alternatively, you can use a terminal to add the variable:

.. code-block:: console

    $> set PATH=%PATH%;<Your Git Path>

In case of any problems with locating of git on your device, `this StackOverflow discussion
(external link) <https://stackoverflow.com/questions/11928561/where-is-git-exe-located>`_
might help.

After this, git should work as expected.

I cannot login to gitlab, but Windows doesn't ask for new login information
--------------------------------------------------------------------------------
Git uses the Windows credential manager and sometimes does not recognize correctly when
a wrong password or username is provided. This leads to errors regarding incorrect login
information but git does not ask for corrected information.

You can force resetting the git credentials in Windows:

- open "Credential Manager" in system control
- Delete information about the git server you are trying to login to
  (in  case of eta_utility: git.rwth-aachen.de)

.. _dymola_license_not_found:

FMU model initialization (Dymola) fails with "Exception: Failed to instantiate model"
-------------------------------------------------------------------------------------
If the log before this also shows the following error, there is a problem with the Dymola
license file.

.. code-block:: console

    [FATAL] The license file was not found. Use the environment variable "DYMOLA_RUNTIME_LICENSE"
    to specify your Dymola license file."

There are multiple options, why this occurs: either the file is not specified at all or
the license server could not be found. To solve the problem, open Windows PowerShell and enter

.. code-block:: console

    dir env:

If the DYMOLA_RUNTIME_LICENSE variable is shown, make sure that it corresponds to the value shown
in Dymola. To do this, open Dymola and go to "Tools" > "License Setup" > "Setup" and read the
value from the field "Local license file, File name".

If the two values are not equal or the variable DYMOLA_RUNTIME_LICENSE does not exist yet,
enter the following command in PowerShell (replacing <File Name> with the value from Dymola):

.. code-block:: console

    [System.Environment]::SetEnvironmentVariable('DYMOLA_RUNTIME_LICENSE','<File Name>')

In case Dymola also does not start, shows an error or starts in trial mode, make sure that
you can connect to the license server correctly. This requires being in the same network as the
server (either physically or using the VPN).

.. _sphinx_not_found:

I want to execute sphinx, but it complains it's not installed
--------------------------------------------------------------

If you receive the following message, you might  have to add the sphinx-scripts to your
environment variables.

.. figure:: figures/dev_02_SphinxError.png
    :width: 470
    :alt: error during documentation build

    Error message shown when trying to build the documentation.

First, find the path to your sphinx installation, then add it via the terminal or through the
"Advanced system settings" in Windows.

.. code-block:: console

    $> set PATH=%PATH%;<Your sphinx path>

.. figure:: figures/dev_03_AddingSphinx2PATH.png
    :width: 470
    :alt: how add to PATH

    Adding sphinx scripts to PATH
