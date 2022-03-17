========================
PDF Federation of Agents
========================

.. image:: https://img.shields.io/travis/maffettone/federation-of-agents.svg
        :target: https://travis-ci.org/maffettone/federation-of-agents

.. image:: https://img.shields.io/pypi/v/federation-of-agents.svg
        :target: https://pypi.python.org/pypi/federation-of-agents


Convenience library to load and run federation of AI agents during PDF experiments.

* Free software: 3-clause BSD license
* Documentation: (COMING SOON!) https://maffettone.github.io/federation-of-agents.

Features
--------

* TODO

User Instructions
-----------------
At the moment some of the requirements are not available on PyPi. Please use a system python > 3.8.
Install from github and set up environment::

    $ python3.8 -m venv agent_env
    $ source agent_env/bin/activate
    $ python -m pip install --upgrade pip wheel
    $ git clone https://github.com/maffettone/xca
    $ cd xca
    $ python -m pip install -e .
    $ cd ../
    $ git clone https://github.com/maffettone/constrained-matrix-factorization
    $ cd constrained-matrix-factorization
    $ python -m pip install -e .
    $ cd ../
    $ git clone https://github.com/NSLS-II-PDF/federation-of-agents
    $ cd federation-of-agents
    $ python -m pip install -e .

Developer Instructions
----------------------
The same instructions as above apply, except all of the `pip install` lines should be replaced by  ::

$ python -m pip install -e . -r requirements-dev.txt
And concluded by::

$ pre-commit install
