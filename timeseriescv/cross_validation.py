import itertools as itt
import numpy as np
import pandas as pd

from abc import abstractmethod
from typing import Iterable, Tuple, List

# fork from https://github.com/sam31415/timeseriescv

D, N, U = -1, 0, 1


class BaseTimeSeriesCrossValidator:
    """
    Abstract class for time series cross-validation.
    Time series cross-validation requires each sample has a prediction time pred_time,
    at which the features are used to predict the response,
    and an evaluation time eval_time, at which the response is known and the error can be computed.
    Importantly, it means that unlike in standard sklearn cross-validation,
    the samples X, response y,
    pred_times and eval_times must all be pandas dataframe/series having the same index.
    It is also assumed that the
    samples are time-ordered with respect to the prediction time (i.e. pred_times is non-decreasing).

    Parameters
    ----------
    n_splits : int, default=10
        Number of folds. Must be at least 2.

    """

    def __init__(self, n_splits=10):
        n_splits = int(n_splits)
        self.n_splits = n_splits
        self.pred_times = None
        self.eval_times = None
        self.indices = None

    @abstractmethod
    def split(self, X: pd.DataFrame):
        self.indices = np.arange(X.shape[0])
        self.eval_times = pd.Series(X.index)
        self.pred_times = pd.Series(X.index)


class CombPurgedKFoldCV(BaseTimeSeriesCrossValidator):
    """
    Purged and embargoed combinatorial cross-validation

    As described in Advances in financial machine learning, Marcos Lopez de Prado, 2018.

    The samples are decomposed into n_splits folds containing equal numbers of samples, without shuffling.
    In each cross validation round, n_test_splits folds are used as the test set,
    while the other folds are used as the train set.
    There are as many rounds as n_test_splits folds among the n_splits folds.

    Each sample should be tagged with a prediction time pred_time and an evaluation time eval_time.
    The split is such that the intervals [pred_times, eval_times] associated to samples
    in the train and test set do not overlap.
    (The overlapping samples are dropped.) In addition, an "embargo" period is defined, giving the minimal time between an
    evaluation time in the test set and a prediction time in the training set. This is to avoid, in the presence of
    temporal correlation, a contamination of the test set by the train set.

    Parameters
    ----------
    n_splits : int, default=10
        Number of folds. Must be at least 2.

    n_test_splits : int, default=2
        Number of folds used in the test set. Must be at least 1.

    embargo_td : pd.Timedelta, default=0
        Embargo period (see explanations above).

    """

    def __init__(
        self, n_splits=10, n_test_splits=2, embargo_td=pd.Timedelta(minutes=0)
    ):
        super().__init__(n_splits)
        n_test_splits = int(n_test_splits)
        self.n_test_splits = n_test_splits
        self.embargo_td = embargo_td

    def split(self, X: pd.DataFrame) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield the indices of the train and test sets.

        Although the samples are passed in the form of a pandas dataframe,
        the indices returned are position indices, not labels.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), required
            Samples. Only used to extract n_samples.

        y : pd.Series, not used, inherited from _BaseKFold


        Returns
        -------
        train_indices: np.ndarray
            A numpy array containing all the indices in the train set.

        test_indices : np.ndarray
            A numpy array containing all the indices in the test set.

        """
        super().split(X)

        # Fold boundaries
        fold_bounds = [
            (fold[0], fold[-1] + 1)
            for fold in np.array_split(self.indices, self.n_splits)
        ]
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

    def compute_train_set(
        self, test_fold_bounds: List[Tuple[int, int]], test_indices: np.ndarray
    ) -> np.ndarray:
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
        # But we now have to purge and embargo
        for test_fold_start, test_fold_end in test_fold_bounds:
            # Purge
            train_indices = purge(self, train_indices, test_fold_start, test_fold_end)
            # Embargo
            train_indices = embargo(self, train_indices, test_indices, test_fold_end)
        return train_indices

    def compute_test_set(
        self, fold_bound_list: List[Tuple[int, int]]
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
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
            test_indices = np.union1d(
                test_indices, self.indices[fold_start:fold_end]
            ).astype(int)
        return test_fold_bounds, test_indices


def embargo(
    cv: BaseTimeSeriesCrossValidator,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    test_fold_end: int,
) -> np.ndarray:
    """
    Apply the embargo procedure to part of the train set.

    This amounts to dropping the train set samples whose prediction time occurs
    within self.embargo_dt of the test set sample evaluation times. This method
    applies the embargo only to the part of the training set immediately following
    the end of the test set determined by test_fold_end.

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
    if not hasattr(cv, "embargo_td"):
        raise ValueError(
            "The passed cross-validation object should have a member cv.embargo_td defining the embargo"
            "time."
        )
    last_test_eval_time = cv.eval_times.iloc[test_indices[:test_fold_end]].max()
    min_train_index = len(
        cv.pred_times[cv.pred_times <= last_test_eval_time + cv.embargo_td]
    )
    if min_train_index < cv.indices.shape[0]:
        allowed_indices = np.concatenate(
            (cv.indices[:test_fold_end], cv.indices[min_train_index:])
        )
        train_indices = np.intersect1d(train_indices, allowed_indices)
    return train_indices


def purge(
    cv: BaseTimeSeriesCrossValidator,
    train_indices: np.ndarray,
    test_fold_start: int,
    test_fold_end: int,
) -> np.ndarray:
    """
    Purge part of the train set.

    Given a left boundary index test_fold_start of the test set,
    this method removes from the train set all the samples whose evaluation time
    is posterior to the prediction time of the first test sample after the boundary.

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
    train_indices_1 = np.intersect1d(
        train_indices, cv.indices[cv.eval_times < time_test_fold_start]
    )
    # The train indices after the end of the test fold.
    train_indices_2 = np.intersect1d(train_indices, cv.indices[test_fold_end:])
    return np.concatenate((train_indices_1, train_indices_2))


def evaluate(
    x,
    y,
    label,
    model,
    lossfunc,
    n_splits=6,
    n_test_splits=2,
    embargo_td=pd.Timedelta(minutes=10),
):
    """
    Args:
        x (pd.DataFrame)            : data of features
        y (pd.DataFrame)            : data of labels
        label (pd.DataFrame)        : answer labels.
        model                       : emsemble model
        n_splits (int)              : the number of groups
                                      default 6.
        n_test_splits (int)         : the number of test groups
                                      default 2.
        embargo_td (pd.Timedelta)   : Embargo time.
                                      Embargo is a loss between current time and observation time.
                                      default pd.Timedelta(minutes=10).
        lossfunc                    : loss function.
    """

    cv = CombPurgedKFoldCV(
        n_splits=n_splits, n_test_splits=n_test_splits, embargo_td=embargo_td
    )

    losses = []

    for train_set, test_set in cv.split(x):

        train_x = x.iloc[train_set]
        train_y = y.iloc[train_set]
        test_x = x.iloc[test_set]
        test_y = y.iloc[test_set]

        model.fit(train_x.values, train_y.values.ravel())
        prob = model.transform(test_x)

        preds = test_y.copy()
        preds.loc[:, "up"] = prob[:, 2]
        preds.loc[:, "neutral"] = prob[:, 1]
        preds.loc[:, "down"] = prob[:, 0]

        preds.loc[:, "label_pred"] = (
            np.argmax(preds[["down", "neutral", "up"]].values, axis=1) - 1
        )

        preds.loc[:, "label_diff"] = label.label_diff
        preds.loc[:, "label_res"] = label.label_res
        preds.loc[:, "pl"] = label.label_diff
        preds.loc[preds.label_pred == D, "pl"] *= -1
        preds.loc[preds.label_pred == N, "pl"] *= 0

        losses.append(lossfunc(preds))
    return losses

