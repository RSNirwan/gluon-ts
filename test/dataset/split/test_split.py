# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import numpy as np
import pandas as pd
import pytest

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.split import DateSplitter, OffsetSplitter
from gluonts.dataset.split.splitter import TimeSeriesSlice


@pytest.fixture
def my_list_data():
    N, T = 2, 10
    target = np.cumsum(np.ones(shape=(N, T)), axis=1)
    s1, s2 = pd.Period("2021-01-01"), pd.Period("2021-01-08")
    data = [{"start": s1, "target": list(t)} for t in target]
    data.append({"start": s2, "target": list(range(3))})  # shorter ts
    return ListDataset(data, freq="1D")


def make_series(data, start="2020", freq="D"):
    index = pd.period_range(start=start, freq=freq, periods=len(data))
    return pd.Series(data, index=index)


def test_ts_slice_to_item():

    sl = TimeSeriesSlice(
        target=make_series(range(100)),
        item="",
        feat_static_cat=[1, 2, 3],
        feat_static_real=[0.1, 0.2, 0.3],
        feat_dynamic_cat=[make_series(range(100))],
        feat_dynamic_real=[make_series(range(100))],
    )

    sl.to_data_entry()


def test_splitter():
    dataset = get_dataset("m4_hourly")
    prediction_length = dataset.metadata.prediction_length
    splitter = DateSplitter(
        prediction_length=prediction_length,
        split_date=pd.Period("1750-01-05 04:00:00", freq="h"),
    )
    train, validation = splitter.split(dataset.train)
    assert len(train[1][0][FieldName.TARGET]) + prediction_length == len(
        validation[1][0][FieldName.TARGET]
    )

    max_history = 2 * prediction_length
    splitter = OffsetSplitter(
        prediction_length=prediction_length,
        split_offset=4 * prediction_length,
        max_history=max_history,
    )
    train, validation = splitter.split(dataset.train)
    assert len(validation[1][0][FieldName.TARGET]) == max_history
    assert len(train[1][0][FieldName.TARGET]) == 4 * prediction_length

    train, validation = splitter.rolling_split(dataset.train, windows=3)
    for i in range(3):
        assert len(validation[1][i][FieldName.TARGET]) == max_history
        assert len(train[1][i][FieldName.TARGET]) == 4 * prediction_length

    max_history = 2 * prediction_length
    splitter = DateSplitter(
        prediction_length=prediction_length,
        split_date=pd.Period("1750-01-05 04:00:00", freq="h"),
        max_history=max_history,
    )
    train, validation = splitter.split(dataset.train)
    assert len(validation[1][0][FieldName.TARGET]) == max_history

    train, validation = splitter.rolling_split(dataset.train, windows=3)
    for i in range(3):
        assert len(validation[1][i][FieldName.TARGET]) == max_history


def test_split_mult_freq():
    splitter = DateSplitter(
        prediction_length=1, split_date=pd.Period("2021-01-01", "2h")
    )

    splitter.split(
        [
            {
                "item_id": "1",
                "target": pd.Series([0, 1, 2]),
                "start": pd.Period("2021-01-01", freq="2H"),
            }
        ]
    )


def test_negative_offset_splitter():
    dataset = ListDataset(
        [
            {"item_id": 0, "start": "2021-03-04", "target": [1.0] * 100},
            {"item_id": 1, "start": "2021-03-04", "target": [2.0] * 50},
        ],
        freq="D",
    )

    split = OffsetSplitter(prediction_length=7, split_offset=-7).split(dataset)

    assert [len(t["target"]) for t in split.train] == [93, 43]
    assert [len(t["target"]) for t in split.test] == [100, 50]

    rolling_split = OffsetSplitter(
        prediction_length=7, split_offset=-21
    ).rolling_split(dataset, windows=3)

    assert [len(t["target"]) for t in rolling_split.train] == [79, 29]
    assert [len(t["target"]) for t in rolling_split.test] == [
        86,
        93,
        100,
        36,
        43,
        50,
    ]


def test_split_ignore_short_ts(my_list_data):
    pl = 2
    splitter = OffsetSplitter(prediction_length=pl, split_offset=-pl)
    split = splitter.split(my_list_data, ignore_short_ts=True)
    train, test = split.train, split.test
    assert len(train) == 3
    assert len(test) == 3
    assert (
        len(next(iter(test))["target"])
        == len(next(iter(train))["target"]) + pl
    )


def test_split_ignore_short_ts_fail(my_list_data):
    pl, num_roll = 2, 2
    splitter = OffsetSplitter(
        prediction_length=pl, split_offset=-num_roll * pl
    )
    with pytest.raises(AssertionError):
        splitter.rolling_split(
            my_list_data, windows=num_roll, ignore_short_ts=False
        )
