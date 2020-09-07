import sys

sys.path.append("../../")
from timeseriescv.cross_validation import CombPurgedKFoldCV as CPCV
import pandas as pd
import numpy as np

periods = 7 * 24 * 60
tidx = pd.date_range("2016-07-01", periods=periods, freq="T")
np.random.seed([3, 1415])
data = np.random.randn(periods)
df = pd.Series(data=data, index=tidx, name="HelloTimeSeries")


cpcv = CPCV(
    n_splits=6,
    n_test_splits=2,
    embargo_td=pd.Timedelta(minutes=30),
    embargo_before_td=pd.Timedelta(minutes=60),
)


for (train_set, test_set) in cpcv.split(df):
    train_X = df.iloc[train_set]
    test_X = df.iloc[test_set]
