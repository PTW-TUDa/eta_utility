.. _faq:

Frequently Asked Questions
==============================

This is a collection of some questions that have been asked frequently.

Model initialization fails with the "Exception: Failed to instantiate model"
------------------------------------------------------------------------------
If the log before this also shows the error:
"[FATAL] The license file was not found. Use the environment variable "DYMOLA_RUNTIME_LICENSE" to specify your
Dymola license file." then there is a problem with the Dymola license file. Either the file is not specified
at all or the license server could not be found. Follow these steps to solve the problem:

- First open Windows PowerShell and enter

    dir env:
- If the DYMOLA_RUNTIME_LICENSE variable is shown, make sure that it corresponds to the value shown in Dymola. To
  do this, open Dymola and go to "Tools" > "License Setup" > "Setup" and read the value from the field
  "Local license file, File name".

- If the two values are not equal or the variable DYMOLA_RUNTIME_LICENSE does not exist yet, enter the following
  command in PowerShell (replacing <File Name> with the value from Dymola:

    [System.Environment]::SetEnvironmentVariable('DYMOLA_RUNTIME_LICENSE','<File Name>')

- In case Dymola also does not start, shows an error or starts in trial mode, make sure that you can connect
  to the license server correctly. This requires being in the same network as the server (either physically or using
  the VPN).
