import itertools as itt
import numbers
import numpy as np
import pandas as pd
import sklearn

from abc import abstractmethod
from typing import Iterable, Tuple, List, Set, Callable


class BaseTimeSeriesCrossValidator():
    """
    Abstract class for time series cross-validation.
    """
    def __init__(self, n_splits=10):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError(f"The number of folds must be of Integral type. {n_split} of type {type(n_splits)}"
                             f" was passed.")
        n_splits = int(n_splits)
        if n_splits <= 1:
            raise ValueError(f"K-fold cross-validation requires at least one train/test split by setting n_splits = 2 "
                             f"or more, got n_splits = {n_splits}.")
        self.n_splits = n_splits
        self.pred_times = None
        self.eval_times = None
        self.indices = None

    @abstractmethod
    def split(self, X: pd.DataFrame, y: pd.Series = None,
              pred_times: pd.Series = None, eval_times: pd.Series = None):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series):
            raise ValueError('X should be a pandas DataFrame/Series.')
        if not isinstance(y, pd.Series) and y is not None:
            raise ValueError('y should be a pandas Series.')
        if not isinstance(pred_times, pd.Series):
            raise ValueError('pred_times should be a pandas Series.')
        if not isinstance(eval_times, pd.Series):
            raise ValueError('eval_times should be a pandas Series.')
        if y is not None and (X.index == y.index).sum() != len(y):
            raise ValueError('X and y must have the same index')
        if (X.index == pred_times.index).sum() != len(pred_times):
            raise ValueError('X and pred_times must have the same index')
        if (X.index == eval_times.index).sum() != len(eval_times):
            raise ValueError('X and eval_times must have the same index')

        self.pred_times = pred_times
        self.eval_times = eval_times
        self.indices = np.arange(X.shape[0])


class CombPurgedKFold(BaseTimeSeriesCrossValidator):
    """
    Purged and embargoed combinatorial cross-validation

    As described in Advances in financial machine learning, Marcos Lopez de Prado, 2018.

    This cross-validation method is adapted to time-series data, where each sample has a prediction time, at which the
    features are used to predict the response, and an evaluation time, at which the response is known and the error
    can be computed. Importantly, it means that unlike in standard sklearn applications, the samples X, response y,
    pred_times and eval_times must all be pandas dataframe having the same index. It is also assumed that the samples
    are time-ordered with respect to the prediction time (i.e. pred_times is non-decreasing).

    The samples are decomposed into n_splits folds containing equal numbers of samples, without shuffling. In each cross
    validation round, n_test_splits folds are used as the test set, while the other folds are used as the train set.
    There are as many rounds as n_test_splits folds among the n_splits folds.

    Each sample should be tagged with a prediction time and an evaluation time (gathered in the pandas series pred_times
    and eval_times. The split is such that the intervals [pred_times, eval_times] associated to samples in the train and
    test set do not overlap. (The overlapping samples are dropped.) In addition, an "embargo" fraction of the dataset
    time span is defined, giving the minimal time between an evaluation time in the test set and a prediction time in
    the training set. This is to avoid, in the presence of temporal correlation, a contamination of the test set by the
    train set.

    Parameters
    ----------
    n_splits : int, default=10
        Number of folds. Must be at least 2.

    n_test_splits : int, default=2
        Number of folds used in the test set. Must be at least 1.

    embargo : float, default=0.0
        Fraction of the sampling timespan that should be used as embargo period (see explanations above).

    TODO: Tests
    TODO: Implement walk-forward cross-validation.

    """
    def __init__(self, n_splits=10, n_test_splits=2, embargo=0.0):
        super().__init__(n_splits)
        if not isinstance(n_test_splits, numbers.Integral):
            raise ValueError(f"The number of test folds must be of Integral type. {n_test_split} of type "
                             f"{type(n_test_splits)} was passed.")
        n_test_splits = int(n_test_splits)
        if n_test_splits <= 0 or n_test_splits >= self.n_splits:
            raise ValueError(f"K-fold cross-validation requires at least one train/test split by setting "
                             f"n_test_splits between 1 and n_splits, got n_test_splits = {n_test_splits}.")
        self.n_test_splits = n_test_splits
        if not isinstance(embargo, numbers.Real):
            raise ValueError(f"The embargo fraction should be of Real type. {embargo} of type "
                             f"{type(embargo)} was passed.")
        embargo = float(embargo)
        if embargo < 0 or embargo > 1.0:
            raise ValueError(f"embargo should lie between 0 and 1, got embargo = {embargo}.")
        self.embargo_frac = embargo
        self.embargo_dt = None

    def split(self, X: pd.DataFrame, y: pd.Series = None,
              pred_times: pd.Series = None, eval_times: pd.Series = None) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield the indices of the train and test sets.

        Although the samples are passed in the form of a pandas dataframe, the indices returned are position indices,
        not labels.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), required
            Samples. Only used to extract n_samples.

        y : pd.Series, not used, inherited from _BaseKFold

        pred_times : pd.Series, shape (n_samples,), required
            Times at which predictions are made. pred_times.index has to coincide with X.index.

        eval_times : pd.Series, shape (n_samples,), required
            Times at which the response becomes available and the error can be computed. eval_times.index has to
            coincide with X.index.

        Returns
        -------
        train_indices: np.ndarray
            A numpy array containing all the indices in the train set.

        test_indices : np.ndarray
            A numpy array containing all the indices in the test set.

        """
        super().split(X, y, pred_times, eval_times)

        self.embargo_dt = self.embargo_frac*(self.pred_times.max() - self.pred_times.min())

        # Fold boundaries
        fold_bounds = [(fold[0], fold[-1] + 1) for fold in np.array_split(self.indices, self.n_splits)]
        # List of all combinations of n_test_splits folds selected to become test sets
        selected_fold_bounds = list(itt.combinations(fold_bounds, self.n_test_splits))
        # In order for the first round to have its whole test set at the end of the dataset
        selected_fold_bounds.reverse()

        for fold_bound_list in selected_fold_bounds:
            # Computes the bounds of the test set, and the corresponding indices
            test_fold_bounds, test_indices = self.compute_test_set(fold_bound_list)
            # Computes the train set indices
            train_indices = self.compute_train_set(test_fold_bounds, test_indices)

            yield train_indices, test_indices

    def compute_train_set(self, test_fold_bounds: List[Tuple[int, int]], test_indices: np.ndarray) -> np.ndarray:
        """
        Compute the position indices of samples in the train set.

        Parameters
        ----------
        test_fold_bounds : List of tuples of position indices
            Each tuple records the bounds of a block of indices in the test set.

        test_indices : np.ndarray
            A numpy array containing all the indices in the test set.

        Returns
        -------
        train_indices: np.ndarray
            A numpy array containing all the indices in the train set.

        """
        # As a first approximation, the train set is the complement of the test set
        train_indices = np.setdiff1d(self.indices, test_indices)
        # print('train_indices', train_indices)
        # print('test_fold_bounds', test_fold_bounds)
        # But we now have to purge and embargo
        for test_fold_start, test_fold_end in test_fold_bounds:
            # Purge
            train_indices = purge(self, train_indices, test_fold_start, test_fold_end)
            # print('array sel', test_indices[:test_indices.searchsorted(test_fold_end)])
            # Embargo
            train_indices = embargo(self, train_indices, test_indices, test_fold_end)
        return train_indices

    def compute_test_set(self, fold_bound_list: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Compute the indices of the samples in the test set.

        Parameters
        ----------
        fold_bound_list: List of tuples of position indices
            Each tuple records the bounds of the folds belonging to the test set.

        Returns
        -------
        test_fold_bounds: List of tuples of position indices
            Like fold_bound_list, but with the neighboring folds in the test set merged.

        test_indices: np.ndarray
            A numpy array containing the test indices.

        """
        test_indices = np.empty(0)
        test_fold_bounds = []
        for fold_start, fold_end in fold_bound_list:
            # Records the boundaries of the current test split
            if not test_fold_bounds or fold_start != test_fold_bounds[-1][-1]:
                test_fold_bounds.append((fold_start, fold_end))
            # If the current test split is contiguous to the previous one, simply updates the endpoint
            elif fold_start == test_fold_bounds[-1][-1]:
                test_fold_bounds[-1] = (test_fold_bounds[-1][0], fold_end)
            test_indices = np.union1d(test_indices, self.indices[fold_start:fold_end]).astype(int)
        return test_fold_bounds, test_indices


def embargo(cv: BaseTimeSeriesCrossValidator, train_indices: np.ndarray,
            test_indices: np.ndarray, test_fold_end: int) -> np.ndarray:
    """
    Apply the embargo procedure to part of the train set.

    This amounts to dropping the train set samples whose prediction time occurs within self.embargo_dt of the test
    set sample evaluation times. This method applies the embargo only to the part of the training set immediately
    following the end of the test set determined by test_fold_end.

    Parameters
    ----------
    cv: Cross-validation class
        Needs to have the attributes cv.pred_times, cv.eval_times, cv.embargo_dt and cv.indices.

    train_indices: np.ndarray
        A numpy array containing all the indices of the samples currently included in the train set.

    test_indices : np.ndarray
        A numpy array containing all the indices of the samples in the test set.

    test_fold_end : int
        Index corresponding to the end of a test set block.

    Returns
    -------
    train_indices: np.ndarray
        The same array, with the indices subject to embargo removed.

    """
    last_test_eval_time = cv.eval_times.iloc[test_indices[:test_fold_end]].max()
    # print("last_test_eval_time", last_test_eval_time)
    # print('train_indices_temp', train_indices_temp)
    min_train_index = len(cv.pred_times[cv.pred_times < last_test_eval_time + cv.embargo_dt])
    if min_train_index < cv.indices.shape[0]:
        allowed_indices = np.concatenate((cv.indices[:test_fold_end], cv.indices[min_train_index:]))
        train_indices = np.intersect1d(train_indices, allowed_indices)
    return train_indices


def purge(cv: BaseTimeSeriesCrossValidator, train_indices: np.ndarray,
          test_fold_start: int, test_fold_end: int) -> np.ndarray:
    """
    Purge part of the train set.

    Given a left boundary index test_fold_start of the test set, this method removes from the train set all the
    samples whose evaluation time is posterior to the prediction time of the first test sample after the boundary.

    Parameters
    ----------
    cv: Cross-validation class
        Needs to have the attributes cv.pred_times, cv.eval_times and cv.indices.

    train_indices: np.ndarray
        A numpy array containing all the indices of the samples currently included in the train set.

    test_fold_start : int
        Index corresponding to the start of a test set block.

    test_fold_end : int
        Index corresponding to the end of the same test set block.

    Returns
    -------
    train_indices: np.ndarray
        A numpy array containing the train indices purged at test_fold_start.

    """
    time_test_fold_start = cv.pred_times.iloc[test_fold_start]
    # The train indices before the start of the test fold, purged.
    train_indices_1 = np.intersect1d(train_indices, cv.indices[cv.eval_times <= time_test_fold_start])
    # The train indices after the end of the test fold.
    train_indices_2 = np.intersect1d(train_indices, cv.indices[test_fold_end:])
    return np.concatenate((train_indices_1, train_indices_2))