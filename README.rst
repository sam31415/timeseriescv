timeseriescv
------------

This package implements two cross-validation algorithms suitable to evaluate machine learning models based on time series
datasets where each sample is tagged with a prediction time and an evaluation time.

Resources
~~~~~~~~~~

* `A Medium post <https://medium.com/@samuel.monnier/cross-validation-tools-for-time-series-ffa1a5a09bf9>`_  providing some motivation and explaining the cross-validation algorithms implemented here in more detail.

* `Advances in financial machine learning <https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086>`_ by Marcos Lopez de Prado. An excellent book that inspired this package.

* `Github repository <https://github.com/sam31415/timeseriescv/>`_


Installation
~~~~~~~~~~~~

timeseriescv can be installed using pip:

    >>> pip install git+https://github.com/pythagorea1/timeseriescv

Content
~~~~~~~

For now the package contains two main classes handling cross-validation:

* ``PurgedWalkForwardCV``: Walk-forward cross-validation with purging.
* ``CombPurgedKFoldCV``: Combinatorial cross-validation with purging and embargoing.

Quick Start
~~~~~~~~~~~~

    >>> from timeseriescv.cross_validation import CombPurgedKFoldCV as CPCV
    >>> cpcv = CPCV(n_splits=10, n_test_splits=2)
    >>> for (train_set, test_set) in cpcv.split(df):
    >>>     train_X = df.iloc[train_set]
    >>>     test_X = df.iloc[test_set]



Remarks concerning the API
~~~~~~~~~~~~~~~~~~~~~~~~~~

The API is as similar to the scikit-learn API as possible. Like the scikit-learn cross-validation classes, the ``split``
method is a generator that yields a pair of numpy arrays containing the positional indices of the samples in the train
and validation set, respectively. The main differences with the scikit-learn API are:

* The ``split`` method takes as arguments not only the predictor values ``X``, but also the prediction times ``pred_times`` and the evaluation times ``eval_times`` of each sample.
* To stay as close to the scikit-learn API as possible, this data is passed as separate parameters. But in order to ensure that they are properly aligned, ``X``, ``pred_times`` and ``eval_times`` are required to be pandas DataFrames/Series sharing the same index.

Check the docstrings of the cross-validation classes for more information.

