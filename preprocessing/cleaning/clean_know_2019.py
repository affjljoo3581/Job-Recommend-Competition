from typing import Any

import pandas as pd


def clean_know_2019(data: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    data = data.copy()

    # Convert the data type to integer and fill the missing values.
    data.loc[:, "sq1":"sq16"] = data.loc[:, "sq1":"sq16"].astype(int)
    for i in range(1, 34):
        data[f"kq{i}_1"] = data[f"kq{i}_1"].astype(int)
        data[f"kq{i}_2"] = data[f"kq{i}_2"].replace(" ", 0).astype(int)

    def isnumerical(x: Any) -> bool:
        try:
            float(x)
            return True
        except ValueError:
            return False

    # Drop the abnormal rows.
    data = data.loc[data.bq6.map(isnumerical)]

    # Fix the shifted (abnormal) features by checking if any numerical features have
    # text data.
    mask = ~data.bq20.map(isnumerical) & ~data.bq31_2.isnull()
    shifted = data.loc[mask, "bq19":"bq31_3"].shift(1, axis=1)
    data.loc[mask, "bq20":"bq31_3"] = shifted
    data.loc[mask, "bq19"] = data.loc[mask, "bq18_10"].str.get(-1)

    mask = data.bq31_3.isnull() & ~data.bq31_2.isnull()
    shifted = data.loc[mask, "bq21_1":"bq31_3"].shift(1, axis=1)
    data.loc[mask, "bq21_2":"bq31_3"] = shifted
    data.loc[mask, "bq21_1"] = data.loc[mask, "bq20_1"].str.get(-1)

    original_mask = data.bq31_3.isnull()
    while True:
        mask = data.bq31_3.isnull()
        if not mask.any():
            break
        shifted = data.loc[mask, "bq18_10":"bq31_3"].shift(1, axis=1)
        data.loc[mask, "bq19":"bq31_3"] = shifted

    columns = "bq19,bq20,bq21_1,bq21_2,bq21_3"
    data.loc[original_mask, columns.split(",")] = 0

    # Drop the abnormal rows.
    data = data.loc[data.bq25.map(isnumerical)]

    # Convert the data type to integer.
    columns = (
        "bq1,bq2,bq3,bq4,bq5,bq6,bq7,bq8_1,bq8_2,bq8_3,bq9,bq10,bq11,bq12_1,bq12_2,"
        "bq12_3,bq12_4,bq12_5,bq13_1,bq13_2,bq13_3,bq14_1,bq14_2,bq14_3,bq14_4,bq14_5,"
        "bq15,bq16_1,bq16_2,bq16_3,bq16_4,bq16_5,bq17,bq18_1,bq18_2,bq18_3,bq18_4,"
        "bq18_5,bq18_6,bq18_7,bq18_8,bq18_9,bq19,bq20,bq21_1,bq21_2,bq21_3,bq25,bq26,"
        "bq27,bq28"
    )
    data[columns.split(",")] = data[columns.split(",")].astype(int)

    # Fill the missing values and convert to integer.
    columns = "bq5_1,bq28_1,bq28_2,bq29,bq30,bq31_1,bq31_2,bq31_3"
    data[columns.split(",")] = (
        data[columns.split(",")].replace(" ", 0).astype(float).astype(int)
    )

    # Fill the missing values with `없음` word.
    columns = "bq5_2,bq18_10,bq20_1,bq22,bq23,bq24,bq27_1"
    data[columns.split(",")] = data[columns.split(",")].replace(" ", "없음")

    # Merge the certificates to the single column.
    data["bq4_1_accum"] = (
        data["bq4_1a"].str.strip()
        + ","
        + data["bq4_1b"].str.strip()
        + ","
        + data["bq4_1c"].str.strip()
    )
    data["bq4_1_accum"] = data["bq4_1_accum"].str.strip(",").replace("", "없음")

    columns = (
        "sq1,sq2,sq3,sq4,sq5,sq6,sq7,sq8,sq9,sq10,sq11,sq12,sq13,sq14,sq15,sq16,kq1_1,"
        "kq1_2,kq2_1,kq2_2,kq3_1,kq3_2,kq4_1,kq4_2,kq5_1,kq5_2,kq6_1,kq6_2,kq7_1,kq7_2,"
        "kq8_1,kq8_2,kq9_1,kq9_2,kq10_1,kq10_2,kq11_1,kq11_2,kq12_1,kq12_2,kq13_1,"
        "kq13_2,kq14_1,kq14_2,kq15_1,kq15_2,kq16_1,kq16_2,kq17_1,kq17_2,kq18_1,kq18_2,"
        "kq19_1,kq19_2,kq20_1,kq20_2,kq21_1,kq21_2,kq22_1,kq22_2,kq23_1,kq23_2,kq24_1,"
        "kq24_2,kq25_1,kq25_2,kq26_1,kq26_2,kq27_1,kq27_2,kq28_1,kq28_2,kq29_1,kq29_2,"
        "kq30_1,kq30_2,kq31_1,kq31_2,kq32_1,kq32_2,kq33_1,kq33_2,bq1,bq2,bq3,bq4,"
        "bq4_1_accum,bq5,bq5_1,bq5_2,bq6,bq7,bq8_1,bq8_2,bq8_3,bq9,bq10,bq11,bq12_1,"
        "bq12_2,bq12_3,bq12_4,bq12_5,bq13_1,bq13_2,bq13_3,bq14_1,bq14_2,bq14_3,bq14_4,"
        "bq14_5,bq15,bq16_1,bq16_2,bq16_3,bq16_4,bq16_5,bq17,bq18_1,bq18_2,bq18_3,"
        "bq18_4,bq18_5,bq18_6,bq18_7,bq18_8,bq18_9,bq18_10,bq19,bq20,bq20_1,bq21_1,"
        "bq21_2,bq21_3,bq22,bq23,bq24,bq25,bq26,bq27,bq27_1,bq28,bq28_1,bq28_2,bq29,"
        "bq30,bq31_1,bq31_2,bq31_3,knowcode"
    )
    return data.idx, data[[x for x in columns.split(",") if x in data.columns]]
