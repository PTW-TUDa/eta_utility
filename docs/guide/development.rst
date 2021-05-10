.. _development:

Contribution to development
===========================

Prerequisite
############

If you want to perform development work, specify install extension "develop" (or ideally all extensions). For more information, see :ref:`install`.

If you are planning to develop on this package, based on the requirements of another
package, you might want to work directly from a local git repository. To do this, clone
this project into a local directory and install it with the development options. Then
uninstall eta_utility from the other projects virtual environment (or system wide) you are working on:

.. code-block:: console

    pip uninstall eta_utility

Then add the path to your local eta_utility respository to the other projects main.py file, before any imports of
eta_utility:

.. code-block:: console

    sys.path.append("repository_path")

Replace repository_path with the path to your local repository. For example "C:\Users\Someone\eta_utility"





Editing this documentation
##########################


Sphinx is used as a documentation-generator. The relevant files are located in the *utility-functions* ➔ *docs* folder.

Sphinx should be already installed on your device. If not, please execute in cmd:

.. code-block:: console

	pip install sphinx sphinx-autobuild sphinx-rtd-theme


Now you can edit the respective *.rst-files* in the *docs* folder. A simple text editor is sufficient.

A helpful start for learning the syntax can be found `here <https://sublime-and-sphinx-guide.readthedocs.io/en/latest/index.html>`_.

For test purposes the following command can be executed in the directory of the documentation:

.. code-block:: console

	make html

This creates a folder named *_build* (inside the *docs* folder) which allows the HTML pages to be locally previewed. This folder will not be pushed in git. Re-execute this command each time after editing the *.rst-files* to see the change (you have to refresh the HTML page, too).

.. image:: Pictures\\Bild13.png
   :width: 700
   :alt: successful HTML build

.. note::
	If this comes up instead, you probably have to add the sphinx-scripts to your PATH and restart cmd.

	.. image:: Pictures\\Bild14.png
   	   :width: 470
   	   :alt: error during HTML build

	|

	.. figure:: Pictures\\Bild12.png
   	   :scale: 13 %
   	   :alt: how add to PATH

	   Adding sphinx-scripts to PATH
