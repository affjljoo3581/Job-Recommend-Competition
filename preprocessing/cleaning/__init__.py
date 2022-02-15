import pandas as pd

from cleaning.clean_know_2017 import clean_know_2017
from cleaning.clean_know_2018 import clean_know_2018
from cleaning.clean_know_2019 import clean_know_2019
from cleaning.clean_know_2020 import clean_know_2020


def clean_know_data(
    data: pd.DataFrame, data_type: int = 2017
) -> tuple[pd.Series, pd.DataFrame]:
    if data_type == 2017:
        return clean_know_2017(data)
    if data_type == 2018:
        return clean_know_2018(data)
    if data_type == 2019:
        return clean_know_2019(data)
    if data_type == 2020:
        return clean_know_2020(data)
    else:
        raise NotImplementedError(f"data type {data_type} is not supported.")
