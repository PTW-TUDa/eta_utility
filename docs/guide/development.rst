.. _development:

Contribution to development
=============================

If you want to perform development work, specify install option "develop" (or ideally all options). For example:

    pip install -e .[develop]

Always also execute

    pre-commit install

before performing the first commits to the resository. This ensures that all pre-commit scripts will run correctly.

If you are planning to develop on this package, based on the requirements of another
package, you might want to work directly from a local git repository. To do this, clone
this project into a local directory and install it with the development options. Then
uninstall eta_utility from the other projects virtual environment (or system wide) you are working on:

    pip uninstall eta_utility

Then add the path to your local eta_utility respository to the other projects main.py file, before any imports of
eta_utility:

    sys.path.append("repository_path")

Replace repository_path with the path to your local repository. For example "C:\Users\Someone\eta_utility"
