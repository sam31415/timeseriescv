import numpy as np
import pandas as pd
import pandas.util.testing as tm
import unittest

from cross_validation import CombPurgedKFold, BaseTimeSeriesCrossValidator, purge, embargo
from typing import Iterable, Tuple, List
from unittest import TestCase


def create_random_sample_set(n_samples, time_shift, randomize_times):
    # Create artificial data
    tm.K = 3
    tm.N = n_samples
    # Random data frame with an hourly index
    test_df = tm.makeTimeDataFrame(freq='H')
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


class TestCombPurgedKFold(TestCase):
    """
    To test:
    - Test doesn't intersect train
    """

    def test_compute_test_set(self):
        """
        We consider a sample set of size 10 with test folds [2:4], [4:6] and [8:10]. The function should return the
        aggregated bounds [2:6], [8:10], as well as the corresponding test indices.
        """
        fold_bound_list = [(2, 4), (4, 6), (8, 10)]
        result1 = [(2, 6), (8,10)]
        result2 = np.array([2, 3, 4, 5, 8, 9])

        cv = CombPurgedKFold(n_splits=5)
        prepare_cv_object(cv, n_samples=10, time_shift='120m', randomlize_times=False)
        agg_fold_bound_list, test_indices = cv.compute_test_set(fold_bound_list)
        self.assertEqual(result1, agg_fold_bound_list)
        self.assertTrue(np.array_equal(result2, test_indices))


class TestPurge(TestCase):

    def test_traintest(self):
        """
        Generates a 2n sample data set consisting of
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
        Generates a similar sample, but with the test set preceding the train set, which starts at n. No sample should
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
        Generates a 2n sample data set consisting of
        - hourly samples
        - two folds, with a test fold followed by a train fold, starting at sample n
        For the first assert statement, a fixed 119m window between the prediction and the the evaluation times. This
        results in sample n to be embargoed.
        For the second assert statement, the window is set to 120m, causing samples n and n + 1 to be embargoed.
        """
        cv = CombPurgedKFold(n_splits=2, n_test_splits=1)
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
        cv = CombPurgedKFold(n_splits=2, n_test_splits=1)
        n = 6
        test_fold_end = n

        prepare_cv_object(cv, n_samples=2 * n, time_shift='119m', randomlize_times=False)
        cv.embargo_td = pd.Timedelta(minutes=120)
        train_indices = cv.indices[n:]
        test_indices = cv.indices[:n]
        result = cv.indices[n + 3:]

        print(cv.embargo_td)
        print(cv.X)
        print(cv.pred_times)
        print(cv.eval_times)
        print(result)
        print(embargo(cv, train_indices, test_indices, test_fold_end))

        self.assertTrue(np.array_equal(result, embargo(cv, train_indices, test_indices, test_fold_end)))

        prepare_cv_object(cv, n_samples=2 * n, time_shift='120m', randomlize_times=False)
        result = cv.indices[n + 4:]
        self.assertTrue(np.array_equal(result, embargo(cv, train_indices, test_indices, test_fold_end)))


if __name__ == '__main__':
    unittest.main()



