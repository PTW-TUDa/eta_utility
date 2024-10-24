from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from eta_utility.timeseries.dataframes import df_interpolate


class TestInterpolate:
    @pytest.fixture
    def uneven_dataframe(self):
        return pd.DataFrame(
            data=[100, 200, 0, 0, 300, 500],
            index=pd.DatetimeIndex(
                [
                    datetime(2024, 10, 18, 15, 3, 45),
                    datetime(2024, 10, 18, 15, 8, 27),
                    datetime(2024, 10, 18, 15, 15, 50),
                    datetime(2024, 10, 18, 16, 15, 3),
                    datetime(2024, 10, 18, 16, 23, 17),
                    datetime(2024, 10, 18, 16, 28, 29),
                ]
            ),
            columns=["Value"],
        )

    def test_uneven_indices(self, uneven_dataframe):
        df_interpolated = df_interpolate(dataframe=uneven_dataframe, freq=timedelta(minutes=5))

        ref_values = [100, 126.6, 158.01, 22.57] + [0] * 12 + [180.36, 366.03, 500]
        assert (df_interpolated["Value"].round(2) == ref_values).all()

    @pytest.mark.parametrize(("limit_direction", "position"), [("forward", 0), ("backward", -1)])
    def test_limit_direction(self, limit_direction, position, uneven_dataframe):
        df_interpolated = df_interpolate(
            dataframe=uneven_dataframe, freq=timedelta(minutes=5), limit_direction=limit_direction
        )
        assert np.isnan(df_interpolated.iloc[position]["Value"])

    @pytest.fixture
    def zero_dataframe(self):
        hours = 5
        index_1 = pd.date_range("1/1/2000", periods=hours, freq="h")  # 1. Jan
        index_2 = pd.date_range("1/2/2000", periods=hours, freq="h")  # 2. Jan
        index_1 = index_1.append(pd.DatetimeIndex([datetime(2000, 1, 1, 15, 43, 56), datetime(2000, 1, 1, 22, 2, 19)]))

        values = [i + 10 for i in range(hours)] + [0, 0] + [i + 10 + hours for i in range(hours)]

        return pd.DataFrame(data=values, index=index_1.append(index_2), columns=["Value"])

    def test_zero_fill(self, zero_dataframe):
        df_interpolated = df_interpolate(dataframe=zero_dataframe, freq=timedelta(hours=1))

        assert df_interpolated.index.is_unique

        zero_values = df_interpolated.iloc[16:23]  # 16 o'clock to 22 o'clock

        assert (zero_values["Value"] == 0).all()
