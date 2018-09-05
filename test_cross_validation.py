import numpy as np
import pandas as pd
import pandas.util.testing as tm
import unittest

try:
    from cross_validation import (BaseTimeSeriesCrossValidator, PurgedWalkForwardCV, CombPurgedKFoldCV, purge, embargo,
                                compute_fold_bounds)
except:
    pass
try:
    from .cross_validation import (BaseTimeSeriesCrossValidator, PurgedWalkForwardCV, CombPurgedKFoldCV, purge, embargo,
                                compute_fold_bounds)
except:
    pass
from typing import Iterable, Tuple, List
from unittest import TestCase


def create_random_sample_set(n_samples, time_shift='120m', randomize_times=False, freq='60T'):
    # Create artificial data
    tm.K = 3
    tm.N = n_samples
    # Random data frame with an hourly index
    test_df = tm.makeTimeDataFrame(freq=freq)
    # Turn the index into a column labeled 'index'
    test_df = test_df.reset_index()
    if randomize_times:
        tm.K = 1
        # Subtract and adds random time deltas to the index column, to create the prediction and evaluation times
        rand_fact = tm.makeDataFrame().reset_index(drop=True).squeeze().iloc[:len(test_df)].abs()
        test_df['index'] = test_df['index'].subtract(rand_fact.apply(lambda x: x * pd.Timedelta(time_shift)))
        rand_fact = tm.makeDataFrame().reset_index(drop=True).squeeze().iloc[:len(test_df)].abs()
        test_df['index2'] = test_df['index'].add(rand_fact.apply(lambda x: x * pd.Timedelta(time_shift)))
    else:
        test_df['index2'] = test_df['index'].apply(lambda x: x + pd.Timedelta(time_shift))
    # Sort the data frame by prediction time
    test_df = test_df.sort_values('index')
    X = test_df[['A', 'B', 'C']]
    pred_times = test_df['index']
    exit_times = test_df['index2']
    return X, pred_times, exit_times


def prepare_cv_object(cv: BaseTimeSeriesCrossValidator, n_samples: int, time_shift: str, randomlize_times: bool):
    X, pred_times, eval_times = create_random_sample_set(n_samples=n_samples, time_shift=time_shift,
                                                         randomize_times=randomlize_times)
    cv.X = X
    cv.pred_times = pred_times
    cv.eval_times = eval_times
    cv.indices = np.arange(X.shape[0])


def prepare_time_inhomogeneous_cv_object(cv: BaseTimeSeriesCrossValidator):
    """
    Creates a sample set consisting in 11 samples at 2h intervals, spanning 20h, as well as 10 samples at 59m intervals,
    with the first samples of each group occurring at the same time.

    pred_times and eval_times have the following values:
                pred_times          eval_times
    0  2000-01-01 00:00:00 2000-01-01 01:00:00
    1  2000-01-01 00:00:00 2000-01-01 01:00:00
    2  2000-01-01 00:59:00 2000-01-01 01:59:00
    3  2000-01-01 01:58:00 2000-01-01 02:58:00
    4  2000-01-01 02:00:00 2000-01-01 03:00:00
    5  2000-01-01 02:57:00 2000-01-01 03:57:00
    6  2000-01-01 03:56:00 2000-01-01 04:56:00
    7  2000-01-01 04:00:00 2000-01-01 05:00:00
    8  2000-01-01 04:55:00 2000-01-01 05:55:00
    9  2000-01-01 05:54:00 2000-01-01 06:54:00
    10 2000-01-01 06:00:00 2000-01-01 07:00:00
    11 2000-01-01 06:53:00 2000-01-01 07:53:00
    12 2000-01-01 07:52:00 2000-01-01 08:52:00
    13 2000-01-01 08:00:00 2000-01-01 09:00:00
    14 2000-01-01 08:51:00 2000-01-01 09:51:00
    15 2000-01-01 10:00:00 2000-01-01 11:00:00
    16 2000-01-01 12:00:00 2000-01-01 13:00:00
    17 2000-01-01 14:00:00 2000-01-01 15:00:00
    18 2000-01-01 16:00:00 2000-01-01 17:00:00
    19 2000-01-01 18:00:00 2000-01-01 19:00:00
    20 2000-01-01 20:00:00 2000-01-01 21:00:00
    """
    X1, pred_times1, eval_times1 = create_random_sample_set(n_samples=11, time_shift='1H', freq='2H')
    X2, pred_times2, eval_times2 = create_random_sample_set(n_samples=10, time_shift='1H', freq='59T')
    data1 = pd.concat([X1, pred_times1, eval_times1], axis=1)
    data2 = pd.concat([X2, pred_times2, eval_times2], axis=1)
    data = pd.concat([data1, data2], axis=0, ignore_index=True)
    data = data.sort_values(by=data.columns[3])
    data = data.reset_index(drop=True)
    X = data.iloc[:, 0:3]
    pred_times = data.iloc[:, 3]
    eval_times = data.iloc[:, 4]

    cv.X = X
    cv.pred_times = pred_times
    cv.eval_times = eval_times
    cv.indices = np.arange(X.shape[0])


class TestPurgedWalkForwardCV(TestCase):
    def test_split(self):
        """
        Apply split to the sample described in the docstring of prepare_time_inhomogeneous_cv_object with n_splits = 5.
        Inspection shows that the pairs test-train sets should respectively be
        1. Train: [0 : 12], test: [13 : 16] (Sample 12 purged from the train set.)
        2. Train: [0 : 16], test: [16, 17]
        3. Train: [0 : 18], test: [18 : 21]
        """
        cv = PurgedWalkForwardCV(n_splits=5)
        prepare_time_inhomogeneous_cv_object(cv)
        count = 0
        for train_set, test_set in cv.split(cv.X, pred_times=cv.pred_times, eval_times=cv.eval_times,
                                            split_by_time=True):
            count += 1
            if count == 1:
                result_train = np.arange(12)
                result_test = np.arange(13, 16)
                self.assertTrue(np.array_equal(result_train, train_set))
                self.assertTrue(np.array_equal(result_test, test_set))
            if count == 2:
                result_train = np.arange(16)
                result_test = np.arange(16, 18)
                self.assertTrue(np.array_equal(result_train, train_set))
                self.assertTrue(np.array_equal(result_test, test_set))
            if count == 3:
                result_train = np.arange(18)
                result_test = np.arange(18, 21)
                self.assertTrue(np.array_equal(result_train, train_set))
                self.assertTrue(np.array_equal(result_test, test_set))


class TestCombPurgedKFoldCV(TestCase):

    def test_split(self):
        """
        Apply split to the sample described in the docstring of prepare_time_inhomogeneous_cv_object, with n_splits = 4
        and n_test_splits = 2. The folds are [0 : 6], [6 : 11], [11 : 16], [16 : 21]. We use an embargo of zero.
        Inspection shows that the pairs test-train sets should respectively be
        [...]
        3. Train: folds 1 and 4, samples [0, 1, 2, 3, 4, 16, 17, 18, 19, 20]. Test: folds 2 and 3, samples [6, 7, 8, 9,
         10, 11, 12, 13, 14, 15]. Sample 5 is purged from the train set.
        4. Train: folds 2 and 3, samples [7, 8, 9, 10, 11, 12, 13, 14, 15]. Test: folds 1 and 4, samples [0, 1, 2, 3, 4,
         5, 16, 17, 18, 19, 20]. Sample 6 is embargoed.
        [...]
        """
        cv = CombPurgedKFoldCV(n_splits=4, n_test_splits=2)
        prepare_time_inhomogeneous_cv_object(cv)
        count = 0
        for train_set, test_set in cv.split(cv.X, pred_times=cv.pred_times, eval_times=cv.eval_times):
            count += 1
            if count == 3:
                result_train = np.array([0, 1, 2, 3, 4, 16, 17, 18, 19, 20])
                result_test = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
                self.assertTrue(np.array_equal(result_train, train_set))
                self.assertTrue(np.array_equal(result_test, test_set))
            if count == 4:
                result_train = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15])
                result_test = np.array([0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20])
                self.assertTrue(np.array_equal(result_train, train_set))
                self.assertTrue(np.array_equal(result_test, test_set))

    def test_compute_test_set(self):
        """
        We consider a sample set of size 10 with test folds [2:4], [4:6] and [8:10]. The function should return the
        aggregated bounds [2:6], [8:10], as well as the corresponding test indices.
        """
        fold_bound_list = [(2, 4), (4, 6), (8, 10)]
        result1 = [(2, 6), (8, 10)]
        result2 = np.array([2, 3, 4, 5, 8, 9])

        cv = CombPurgedKFoldCV(n_splits=5)
        prepare_cv_object(cv, n_samples=10, time_shift='120m', randomlize_times=False)
        agg_fold_bound_list, test_indices = cv.compute_test_set(fold_bound_list)
        self.assertEqual(result1, agg_fold_bound_list)
        self.assertTrue(np.array_equal(result2, test_indices))


class TestComputeFoldBounds(TestCase):
    def test_by_samples(self):
        """
        Use a 10 sample set, with 5 folds. The fold left bounds are at 0, 2, 4, 6, and 8.
        """
        cv = PurgedWalkForwardCV(n_splits=5)
        prepare_cv_object(cv, n_samples=10, time_shift='120m', randomlize_times=False)
        result = [0, 2, 4, 6, 8]
        self.assertEqual(result, compute_fold_bounds(cv, False))

    def test_by_time(self):
        """
        Create a sample set as described in the docstring of prepare_time_inhomogeneous_cv_object. Inspection shows
        that the fold left bounds are at 0, 7, 13, 16, 18.
        """
        cv = PurgedWalkForwardCV(n_splits=5)
        prepare_time_inhomogeneous_cv_object(cv)
        result = [0, 7, 13, 16, 18]
        self.assertTrue(all(result[i] == compute_fold_bounds(cv, True)[i] for i in range(5)))


class TestPurge(TestCase):

    def test_traintest(self):
        """
        Generate a 2n sample data set consisting of
        - hourly samples
        - two folds, with a train fold followed by a test fold, starting at sample n + 1
        For the first assert statement, a fixed 119m window between the prediction and the the evaluation times. This
        results in sample n to be purged.
        For the second assert statement, as 120m window is chosen, resulting in samples n - 1 and n to be purged.
        """
        cv = BaseTimeSeriesCrossValidator(n_splits=2)
        n = 6
        test_fold_start = n + 1
        test_fold_end = 2 * n

        prepare_cv_object(cv, n_samples=2 * n, time_shift='119m', randomlize_times=False)
        train_indices = cv.indices[:n]
        result = cv.indices[0:n]
        self.assertTrue(np.array_equal(result, purge(cv, train_indices, test_fold_start, test_fold_end)))

        prepare_cv_object(cv, n_samples=2 * n, time_shift='120m', randomlize_times=False)
        result = cv.indices[0:n - 1]
        self.assertTrue(np.array_equal(result, purge(cv, train_indices, test_fold_start, test_fold_end)))

    def test_testtrain(self):
        """
        Generate a similar sample, but with the test set preceding the train set, which starts at n. No sample should
        be purged.
        """
        cv = BaseTimeSeriesCrossValidator(n_splits=2)
        n = 6
        test_fold_start = 0
        test_fold_end = n

        prepare_cv_object(cv, n_samples=2 * n, time_shift='120m', randomlize_times=False)
        train_indices = cv.indices[n:]
        result = cv.indices[n:]
        self.assertTrue(np.array_equal(result, purge(cv, train_indices, test_fold_start, test_fold_end)))


class TestEmbargo(TestCase):

    def test_zero_embargo(self):
        """
        Generate a 2n sample data set consisting of
        - hourly samples
        - two folds, with a test fold followed by a train fold, starting at sample n
        For the first assert statement, a fixed 119m window between the prediction and the the evaluation times. This
        results in sample n to be embargoed.
        For the second assert statement, the window is set to 120m, causing samples n and n + 1 to be embargoed.
        """
        cv = CombPurgedKFoldCV(n_splits=2, n_test_splits=1)
        n = 6
        test_fold_end = n

        prepare_cv_object(cv, n_samples=2 * n, time_shift='119m', randomlize_times=False)
        cv.embargo_td = pd.Timedelta(minutes=0)
        train_indices = cv.indices[n:]
        test_indices = cv.indices[:n]
        result = cv.indices[n + 1:]
        self.assertTrue(np.array_equal(result, embargo(cv, train_indices, test_indices, test_fold_end)))

        prepare_cv_object(cv, n_samples=2 * n, time_shift='120m', randomlize_times=False)
        result = cv.indices[n + 2:]
        self.assertTrue(np.array_equal(result, embargo(cv, train_indices, test_indices, test_fold_end)))

    def test_nonzero_embargo(self):
        """
        Same with an embargo delay of 2h. two more samples have to be embargoed in each case.
        """
        cv = CombPurgedKFoldCV(n_splits=2, n_test_splits=1)
        n = 6
        test_fold_end = n

        prepare_cv_object(cv, n_samples=2 * n, time_shift='119m', randomlize_times=False)
        cv.embargo_td = pd.Timedelta(minutes=120)
        train_indices = cv.indices[n:]
        test_indices = cv.indices[:n]
        result = cv.indices[n + 3:]

        self.assertTrue(np.array_equal(result, embargo(cv, train_indices, test_indices, test_fold_end)))

        prepare_cv_object(cv, n_samples=2 * n, time_shift='120m', randomlize_times=False)
        result = cv.indices[n + 4:]
        self.assertTrue(np.array_equal(result, embargo(cv, train_indices, test_indices, test_fold_end)))


if __name__ == '__main__':
    unittest.main()

