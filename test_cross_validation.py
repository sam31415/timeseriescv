import pandas as pd
import pandas.util.testing as tm
import unittest

from unittest import TestCase
from .cross_validation import  CombPurgedKFold


class TestCombPurgedKFold(TestCase):
    """
    To test:
    - Test doesn't intersect train
    - Purge
    - Embargo
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


if __name__ == '__main__':
    unittest.main()
