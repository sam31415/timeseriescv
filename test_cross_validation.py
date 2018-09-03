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

    def setUp(self):
        # Create artificial data
        tm.K = 1
        tm.N = 100
        # Random data frame with an hourly index
        test_df = tm.makeTimeDataFrame(freq='H')
        # Turn the index into a column labeled 'index'
        test_df = test_df.reset_index()
        # Subtract and adds random time deltas to the index column, to create the prediction and evaluation times
        rand_fact = tm.makeDataFrame().reset_index(drop=True).squeeze().iloc[:len(test_df)].abs()
        test_df['index'] = test_df['index'].subtract(rand_fact.apply(lambda x: x * pd.Timedelta('4H')))
        rand_fact = tm.makeDataFrame().reset_index(drop=True).squeeze().iloc[:len(test_df)].abs()
        test_df['index2'] = test_df['index'].add(rand_fact.apply(lambda x: x * pd.Timedelta('4H')))
        # Sort the data frame by prediction time
        test_df = test_df.sort_values('index')
        self.X = test_df['A']
        self.pred_times = test_df['index']
        self.exit_times = test_df['index2']

        # Create the cross-validation class with non-zero embargo
        self.cv = CombPurgedKFold(n_splits=10, n_test_splits=2, embargo=0.02)

    def test_data_equal_indices(self):
        tm.assert_index_equal(self.X.index, self.pred_times.index)
        tm.assert_index_equal(self.X.index, self.exit_times.index)

    def test_data_pred_before_eval(self):
        assert ((self.exit_times.subtract(self.pred_times) < pd.Timedelta('0H')).sum() == 0)


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
        cv = BaseTimeSeriesCrossValidator(n_splits=2)
        n = 6
        test_fold_end = n

        prepare_cv_object(cv, n_samples=2 * n, time_shift='119m', randomlize_times=False)
        cv.embargo_dt = 0 * (cv.pred_times.max() - cv.pred_times.min())
        train_indices = cv.indices[n:]
        test_indices = cv.indices[:n]
        result = cv.indices[n + 1:]
        self.assertTrue(np.array_equal(result, embargo(cv, train_indices, test_indices, test_fold_end)))

        prepare_cv_object(cv, n_samples=2 * n, time_shift='120m', randomlize_times=False)
        result = cv.indices[n + 2:]
        self.assertTrue(np.array_equal(result, embargo(cv, train_indices, test_indices, test_fold_end)))

    def test_nonzero_embargo(self):
        """
        Same with an embargo fraction of 2/(2*n-1), corresponding to an embargo delay of 2h. two more samples have to be
        embargoed in each case.
        """
        cv = BaseTimeSeriesCrossValidator(n_splits=2)
        n = 6
        test_fold_end = n

        prepare_cv_object(cv, n_samples=2 * n, time_shift='119m', randomlize_times=False)
        cv.embargo_dt = 2/(2*n-1) * (cv.pred_times.max() - cv.pred_times.min())
        train_indices = cv.indices[n:]
        test_indices = cv.indices[:n]
        result = cv.indices[n + 3:]
        self.assertTrue(np.array_equal(result, embargo(cv, train_indices, test_indices, test_fold_end)))

        prepare_cv_object(cv, n_samples=2 * n, time_shift='120m', randomlize_times=False)
        result = cv.indices[n + 4:]
        self.assertTrue(np.array_equal(result, embargo(cv, train_indices, test_indices, test_fold_end)))

if __name__ == '__main__':
    unittest.main()



