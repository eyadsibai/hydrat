# Introduction #

hydrat is a framework for performing classification tasks in a research context. It was designed to facilitate comparison between different classifiers, and is actively used in the author's research work.

# Prerequisites #

You will need numpy, scipy and PyTables. You must install these before attempting to install hydrat, as easy\_install/pip has difficulty installing these packages. numpy is in ubuntu precise main (python-numpy), and scipy and PyTables are in ubuntu precise universe (python-scipy and python-tables).

All other prerequisites for hydrat will be resolved by easy\_install/pip.

# Virtualenv #

[virtualenv](http://pypi.python.org/pypi/virtualenv) is a tool to create isolated Python environments. On Ubuntu, you can install it via
```
sudo apt-get install python-virtualenv
```
You are not required to use virtualenv to use hydrat, but it is highly recommended if you plan to develop hydrat. The author also recommends using [virtualenvwrapper](http://www.doughellmann.com/projects/virtualenvwrapper/) to manage virtualenv.

# Installing hydrat #

hydrat is packaged using Python's setuptools, and is distributed via PyPI. This means that for most users, installation can be carried out using easy\_install or pip:

```
easy_install hydrat
```

or

```
pip install hydrat
```

Some users have reported difficulty in using the version of pip distributed with Ubuntu 10.04. If you are affected, the author recommends you use easy\_install or update pip.

For systems where you do not have administrative rights, the author recommends the use of [virtualenv](http://pypi.python.org/pypi/virtualenv).


Besides, you can also download the source code from here (see 'Source' for instructions), and install the hydrat by yourselves.

```
python setup.py install
```

or

```
python setup.py develop
```

# Using the development version #

hydrat is under heavy development, and many times the latest version on pypi will be lagging behind the project source in features and/or bugfixes. To get the latest development sources, you can clone the hydrat repository using the following commands (you will need to have mercurial installed):

```
hg clone https://hydrat.googlecode.com/hg/ hydrat 
cd hydrat
python setup.py install
```

If you plan to make changes to hydrat, you may want to install hydrat in 'develop' mode instead:

```
hg clone https://hydrat.googlecode.com/hg/ hydrat 
cd hydrat
python setup.py develop
```

# Configuration #

hydrat does not implement many classifiers natively. Instead, it relies on interfacing with external packages (the full list is at SupportedClassifiers) to provide the core classification functionality.

In order to access the classifiers, hydrat must be configured. This is done via a configuration file called '.hydratrc'. hydrat looks for '.hydratrc' in two places:
  * your home directory
  * your current working directory

Any options specified in the latter will supersede those specified in the former.

To generate an initial '.hydratrc', you can use the 'configure' command of hydrat's CLI.

```
hydrat configure
```

This will create a '.hydratrc' file in your current working directory.

You will need to edit the .hydratrc file to specify the location of corpora, and some tools that hydrat is not yet able to auto-detect.