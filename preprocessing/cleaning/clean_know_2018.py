from typing import Any

import pandas as pd


def clean_know_2018(data: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    data = data.copy()

    # Convert the data type to integer.
    data.loc[:, "cq1":"iq6"] = data.loc[:, "cq1":"iq6"].astype(int)

    def isnumerical(x: Any) -> bool:
        try:
            float(x)
            return True
        except ValueError:
            return False

    # Fix the shifted (abnormal) features by checking if any numerical features have
    # text data.
    mask = ~data.bq4.map(isnumerical)
    data.loc[mask, "bq4":"bq41_3"] = data.loc[mask, "bq4":"bq41_3"].shift(1, axis=1)
    data.loc[mask, "bq4"] = (data.loc[mask, "bq4_1a"] == " ").map(
        lambda b: "2" if b else "1"
    )

    mask = ~data.bq28.map(isnumerical)
    shifted = data.loc[mask, "bq12_1":"bq41_3"].shift(1, axis=1)
    data.loc[mask, "bq12_1":"bq41_3"] = shifted
    data.loc[mask, "bq12_1"] = " "

    data = data.loc[data.bq37.map(isnumerical) | data.bq41_3.isnull()]

    while True:
        mask = data.bq41_3.isnull()
        if not mask.any():
            break
        shifted = data.loc[mask, "bq28_1":"bq41_3"].shift(1, axis=1)
        data.loc[mask, "bq29":"bq41_3"] = shifted

    # Convert the data type to integer.
    columns = (
        "bq1,bq2,bq3,bq4,bq5,bq6,bq7,bq8_1,bq8_2,bq8_3,bq9,bq10,bq11,bq13,bq14,bq15,"
        "bq16,bq17,bq18,bq19,bq20,bq25,bq26_1,bq26_2,bq26_3,bq26_4,bq27,bq28,bq34,bq35,"
        "bq36,bq37,bq38"
    )
    data[columns.split(",")] = data[columns.split(",")].astype(float).astype(int)

    # Fill the missing values and convert to integer.
    columns = (
        "bq5_1,bq21,bq221,bq222,bq223,bq231,bq232,bq233,bq234,bq235,bq241,bq242,bq243,"
        "bq244,bq245,bq25_1,bq26_1a,bq26_2a,bq26_3a,bq26_4a,bq38_1,bq38_2,bq39,bq40,"
        "bq41_1,bq41_2,bq41_3"
    )
    data[columns.split(",")] = (
        data[columns.split(",")].replace(" ", 0).astype(float).astype(int)
    )

    columns = "bq12_1,bq12_2,bq12_3,bq12_4,bq12_5"
    data[columns.split(",")] = (
        data[columns.split(",")].replace(" ", 9).astype(float).astype(int)
    )

    # Fill the missing values with `없음` word.
    columns = "bq5_2,bq28_1,bq29,bq30,bq31,bq32,bq33,bq37_1"
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
        "cq1,cq2,cq3,cq4,cq5,cq6,cq7,cq8,cq9,cq10,cq11,cq12,cq13,cq14,cq15,cq16,cq17,"
        "cq18,cq19,cq20,cq21,cq22,cq23,cq24,cq25,cq26,cq27,cq28,cq29,cq30,cq31,cq32,"
        "cq33,cq34,cq35,cq36,cq37,cq38,cq39,cq40,cq41,cq42,cq43,cq44,cq45,cq46,cq47,"
        "cq48,cq49,cq50_1,cq50_2,cq50_3,cq50_4,cq50_5,cq50_6,cq50_7,cq50_8,iq1,iq2,iq3,"
        "iq4,iq5,iq6,bq1,bq2,bq3,bq4,bq4_1_accum,bq5,bq5_1,bq5_2,bq6,bq7,bq8_1,bq8_2,"
        "bq8_3,bq9,bq10,bq11,bq12_1,bq12_2,bq12_3,bq12_4,bq12_5,bq13,bq14,bq15,bq16,"
        "bq17,bq18,bq19,bq20,bq21,bq221,bq222,bq223,bq231,bq232,bq233,bq234,bq235,"
        "bq241,bq242,bq243,bq244,bq245,bq25,bq25_1,bq26_1,bq26_1a,bq26_2,bq26_2a,"
        "bq26_3,bq26_3a,bq26_4,bq26_4a,bq27,bq28,bq28_1,bq29,bq30,bq31,bq32,bq33,bq34,"
        "bq35,bq36,bq37,bq37_1,bq38,bq38_1,bq38_2,bq39,bq40,bq41_1,bq41_2,bq41_3,"
        "knowcode"
    )
    return data.idx, data[[x for x in columns.split(",") if x in data.columns]]
