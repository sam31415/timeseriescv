timeseriescv
------------

Package implementing two cross-validation algorithms suitable to evaluate machine learning models based on time series
datasets.

Installation
~~~~~~~~~~~~

timeseriescv can be installed using pip:
    >>> pip install timeseriescv

Content
~~~~~~~

For now the package contains two main classes handling cross-validation:

* ``PurgedWalkForwardCV``: Walk-forward cross-validation with purging.
* ``CombPurgedKFoldCV``: Combinatorial cross-validation with purging and embargoing.

Check their respective docstrings for more information.



