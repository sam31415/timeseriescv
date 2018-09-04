import numpy as np
import pandas as pd
import pandas.util.testing as tm
import unittest

from cross_validation import (BaseTimeSeriesCrossValidator, PurgedWalkForwardCV, CombPurgedKFoldCV, purge, embargo,
                              compute_fold_bounds)
from typing import Iterable, Tuple, List
from unittest import TestCase


def create_random_sample_set(n_samples, time_shift='120m', randomize_times=False, freq='60m'):
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

    #if isinstance(cv, PurgedWalkForwardCV):



# class TestPurgedWalkForwardCV(TestCase):
#     def test_compute_test_set(self):
#         """
#
#         """
#         cv = PurgedWalkForwardCV(n_splits=5)
#         prepare_cv_object(cv, n_samples=10, time_shift='120m', randomlize_times=False)
#         cv.fold_bounds = compute_fold_bounds(cv, split_by_time)
#         pass

class TestCombPurgedKFoldCV(TestCase):

    def test_compute_test_set(self):
        """
        We consider a sample set of size 10 with test folds [2:4], [4:6] and [8:10]. The function should return the
        aggregated bounds [2:6], [8:10], as well as the corresponding test indices.
        """
        fold_bound_list = [(2, 4), (4, 6), (8, 10)]
        result1 = [(2, 6), (8,10)]
        result2 = np.array([2, 3, 4, 5, 8, 9])

        cv = CombPurgedKFoldCV(n_splits=5)
        prepare_cv_object(cv, n_samples=10, time_shift='120m', randomlize_times=False)
        agg_fold_bound_list, test_indices = cv.compute_test_set(fold_bound_list)
        self.assertEqual(result1, agg_fold_bound_list)
        self.assertTrue(np.array_equal(result2, test_indices))


class TestComputeFoldBounds(TestCase):
    def test_by_samples(self):
        """
        Uses a 10 sample set, with 5 folds. The fold left bounds are at 0, 2, 4, 6, and 8.
        """
        cv = PurgedWalkForwardCV(n_splits=5)
        prepare_cv_object(cv, n_samples=10, time_shift='120m', randomlize_times=False)
        result = [0, 2, 4, 6, 8]
        self.assertEqual(result, compute_fold_bounds(cv, False))

    def test_by_time(self):
        """
        Creates a sample set consisting in 11 samples at 2h intervals, spanning 20h, as well as 10 samples at 59m
        intervals, with the first samples of each group occurring at the same time. Inspection shows that the fold left
        bounds are at 0, 7, 13, 16, 18.
        """
        cv = PurgedWalkForwardCV(n_splits=5)

        X1, pred_times1, eval_times1 = create_random_sample_set(n_samples=11, freq='2H')
        X2, pred_times2, eval_times2 = create_random_sample_set(n_samples=10, freq='59T')
        data1 = pd.concat([X1, pred_times1, eval_times1], axis=1)
        data2 = pd.concat([X2, pred_times2, eval_times2], axis=1)
        data = pd.concat([data1, data2], axis=0, ignore_index=True)
        data = data.sort_values(by=data.columns[3])
        X = data.iloc[:, 0:3]
        pred_times = data.iloc[:, 3]
        eval_times = data.iloc[:, 4]

        cv.X = X
        cv.pred_times = pred_times
        cv.eval_times = eval_times
        cv.indices = np.arange(X.shape[0])

        result = [0, 7, 13, 16, 18]
        self.assertTrue(all(result[i] == compute_fold_bounds(cv, True)[i] for i in range(5)))


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



