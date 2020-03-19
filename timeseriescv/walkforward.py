
import itertools as itt
import pandas as pd
import numpy as np
from typing import Iterable, Tuple, List
from core import BaseTimeSeriesCrossValidator, purge, embargo

class PurgedWalkForwardCV(BaseTimeSeriesCrossValidator):
    """
    Purged walk-forward cross-validation

    As described in Advances in financial machine learning, Marcos Lopez de Prado, 2018.

    The samples are decomposed into n_splits folds containing equal numbers of samples, without shuffling. In each cross
    validation round, n_test_splits contiguous folds are used as the test set, while the train set consists in between
    min_train_splits and max_train_splits immediately preceding folds.

    Each sample should be tagged with a prediction time pred_time and an evaluation time eval_time. The split is such
    that the intervals [pred_times, eval_times] associated to samples in the train and test set do not overlap. (The
    overlapping samples are dropped.)

    With split_by_times = True in the split method, it is also possible to split the samples in folds spanning equal
    time intervals (using the prediction time as a time tag), instead of folds containing equal numbers of samples.

    Parameters
    ----------
    n_splits : int, default=10
        Number of folds. Must be at least 2.

    n_test_splits : int, default = 1
        Number of folds used in the test set. Must be at least 1.

    min_train_splits: int, default = 2
        Minimal number of folds to be used in the train set.

    max_train_splits: int, default = None
        Maximal number of folds to be used in the train set. If None, there is no upper limit.

    """
    def __init__(self, n_splits=10, n_test_splits=1, min_train_splits=2, purge_count=3):
        super().__init__(n_splits)
        self.n_test_splits = n_test_splits
        self.min_train_splits = min_train_splits
        self.max_train_splits = self.n_splits - self.n_test_splits
        self.purge_count = purge_count
        self.fold_bounds = []


    def split(self, X: pd.DataFrame, split_by_time: bool = False) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield the indices of the train and test sets.

        Although the samples are passed in the form of a pandas dataframe, the indices returned are position indices,
        not labels.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), required
            Samples. Only used to extract n_samples.

        split_by_time: bool
            If False, the folds contain an (approximately) equal number of samples. If True, the folds span identical
            time intervals.

        Returns
        -------
        train_indices: np.ndarray
            A numpy array containing all the indices in the train set.

        test_indices : np.ndarray
            A numpy array containing all the indices in the test set.

        """
        super().split(X)

        # Fold boundaries
        self.fold_bounds = compute_fold_bounds(self, split_by_time)

        count_folds = 0
        for fold_bound in self.fold_bounds:
            if count_folds < self.min_train_splits:
                count_folds = count_folds + 1
                continue
            if self.n_splits - count_folds < self.n_test_splits:
                break
            # Computes the bounds of the test set, and the corresponding indices
            test_indices = self.compute_test_set(fold_bound, count_folds)
            test_length = len(test_indices)
            # Computes the train set indices
            train_indices = self.compute_train_set(fold_bound, count_folds)
            train_length = len(train_indices)
            train_indices = train_indices[train_length-test_length:]
            test_indices = test_indices[self.purge_count:]

            count_folds = count_folds + 1
            yield train_indices, test_indices


    def compute_train_set(self, fold_bound: int, count_folds: int) -> np.ndarray:
        """
        Compute the position indices of samples in the train set.

        Parameters
        ----------
        fold_bound : int
            Bound between the train set and the test set.

        count_folds : int
            The number (starting at 0) of the first fold in the test set.

        Returns
        -------
        train_indices: np.ndarray
            A numpy array containing all the indices in the train set.

        """
        if count_folds > self.max_train_splits:
            start_train = self.fold_bounds[count_folds - self.max_train_splits]
        else:
            start_train = 0
        train_indices = np.arange(start_train, fold_bound)
        # Purge
        train_indices = purge(self, train_indices, fold_bound, self.indices[-1])
        return train_indices


    def compute_test_set(self, fold_bound: int, count_folds: int) -> np.ndarray:
        """
        Compute the indices of the samples in the test set.

        Parameters
        ----------
        fold_bound : int
            Bound between the train set and the test set.

        count_folds : int
            The number (starting at 0) of the first fold in the test set.

        Returns
        -------
        test_indices: np.ndarray
            A numpy array containing the test indices.

        """
        if self.n_splits - count_folds > self.n_test_splits:
            end_test = self.fold_bounds[count_folds + self.n_test_splits]
        else:
            end_test = self.indices[-1] + 1
        return np.arange(fold_bound, end_test)

def compute_fold_bounds(cv: BaseTimeSeriesCrossValidator, split_by_time: bool) -> List[int]:
    """
    Compute a list containing the fold (left) boundaries.

    Parameters
    ----------
    cv: BaseTimeSeriesCrossValidator
        Cross-validation object for which the bounds need to be computed.
    split_by_time: bool
        If False, the folds contain an (approximately) equal number of samples. If True, the folds span identical
        time intervals.
    """
    if split_by_time:
        full_time_span = cv.pred_times.max() - cv.pred_times.min()
        fold_time_span = full_time_span / cv.n_splits
        fold_bounds_times = [cv.pred_times.iloc[0] + fold_time_span * n for n in range(cv.n_splits)]
        return cv.pred_times.searchsorted(fold_bounds_times)
    else:
        return [fold[0] for fold in np.array_split(cv.indices, cv.n_splits)]

