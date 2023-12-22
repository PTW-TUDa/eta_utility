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

    $ set PATH=%PATH%;<Your Git Path>

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
  (in  case of eta_utility: "git.ptw.maschinenbau.tu-darmstadt.de <git.ptw.maschinenbau.tu-darmstadt.de>"_).

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

    $ set PATH=%PATH%;<Your sphinx path>

.. figure:: figures/dev_03_AddingSphinx2PATH.png
    :width: 470
    :alt: how add to PATH

    Adding sphinx scripts to PATH

.. _julia_not_found:

I want to install Julia, but the Julia executable cannot be found
------------------------------------------------------------------
If you receive the following error message, when you want to install the Julia executable:

.. code-block:: console

    $ Exception: Julia executable cannot be found. If you have installed Julia, make sure Julia executable is in the system path. If you have not installed Julia, download from https://julialang.org/downloads/ and install it.

Add the path from Julia to Windows as described in :ref:`install_julia` and restart eta-utility.

.. _packages_error:

I want to start *eta_utility*, but in some external packages there were changes
----------------------------------------------------------------------------------
If you want to use *eta_utility*, but you get errors in some Python packages, because there were changes in the source \
code, update *eta_utility* with the following command (add extra requirements like `eta_x` as needed):

.. code-block:: console

    $ pip install --upgrade --upgrade-strategy=only-if-needed eta_utility

I want to start a julia experiment, but there is an AttributeError
-------------------------------------------------------------------
If you receive the following (or a similar) error message, when you want to start a julia experiment:

.. code-block:: console

    $ Exception: AttributeError: module 'eta_utility.eta_x.agents' has no attribute 'Nsga2'.

Make sure PyJulia is installed in the correct virtual environment as described in :ref:`install_julia`.

I made changes in the NSGA2 agent and want to do a commit, but my tests are not passing
---------------------------------------------------------------------------------------
Make sure that you have refreshed the stored agent model as described in :ref:`testing_your_code`.

Resolve FMPy compilation issue on macOS (x64)
-----------------------------------------------
If you start developing on macOS, you might encounter errors when compiling FMPy, should the pre-complied binaries not be available for your system.

The error message might look like this:

.. code-block:: console

    $ clang: error: unsupported option '-fopenmp'
    $ error: command 'clang' failed with exit status 1

Here's how you can manually compile the FMU. Make sure to replace all occurrences of "fmu_file" with the actual name of the FMU:

1. Install the `zip` package using Homebrew:

.. code-block:: console

   $ brew install zip


2. Create a folder where the FMU can be extracted and extract the FMU:

.. code-block:: console

   $ [ ! -d fmu_extract ] && mkdir fmu_extract
   $ unzip -u path/to/fmu_file.fmu -d fmu_extract


3. If you encounter an error with compiling the `ModelicaInternal.c` file, insert a function declaration before the call in `/sources/ModelicaInternal.c`, add the following declaration:

.. code-block:: console

   $ int creat(const char *path, mode_t mode) __DARWIN_ALIAS_C(creat);


4. Switch to the `sources` folder:

.. code-block:: console

   $ cd fmu_extract/sources


5. Run the following Clang compiler command with the `-w` flags to clear the output of warnings (replace `/path/to/fmpy/` with the actual path to your fmpy installation):

.. code-block:: console

   $ clang -w -c -arch x86_64 -arch arm64 -I. -I/path/to/fmpy/c-code all.c && clang -w -shared -arch x86_64 -arch arm64 -ofmu_file.dylib *.o -lm


6. Move the output to the `darwin64` folder:

.. code-block:: console

   $ [ ! -d ../binaries/darwin64 ] && mkdir ../binaries/darwin64
   $ mv out.so ../binaries/darwin64/fmu_file.dylib


7. Pack the new FMU, which contains the compiled files:

.. code-block:: console

   $ cd ..
   $ zip -r ../fmu_file.fmu *


8. Clean up:

.. code-block:: console

   $ cd ..
   $ rm -rf fmu_extract


After following these steps, you should have a new FMU file that contains the compiled files and can be used on macOS systems.

This resolution has been tested on macOS Ventura 13.4.1 (xarm64: M1).
See also:
https://git.ptw.maschinenbau.tu-darmstadt.de/eta-fabrik/public/eta-utility/-/issues/200
