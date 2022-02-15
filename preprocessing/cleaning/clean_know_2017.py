import pandas as pd


def clean_know_2017(data: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    data = data.copy()

    # Convert the data type to integer and fill the missing values.
    for i in range(1, 42):
        data[f"aq{i}_1"] = data[f"aq{i}_1"].astype(int)
        data[f"aq{i}_2"] = data[f"aq{i}_2"].replace(" ", 0).astype(int)

    # Convert the data type to integer.
    columns = (
        "bq1,bq2,bq3,bq4,bq6,bq7,bq8_1,bq8_2,bq8_3,bq9,bq10,bq11,bq12_1,bq12_5,bq13,"
        "bq14,bq15_1,bq15_2,bq15_3,bq16,bq17,bq18_1,bq18_2,bq18_3,bq18_4,bq18_5,bq18_6,"
        "bq18_7,bq19,bq20,bq21,bq22,bq24_1,bq24_2,bq24_3,bq24_4,bq24_5,bq24_6,bq24_7,"
        "bq24_8,bq25,bq26,bq27,bq28,bq29,bq35,bq36,bq38,bq39_2"
    )
    data[columns.split(",")] = data[columns.split(",")].astype(int)

    # Fill the missing values and convert to integer.
    columns = "bq5_1,bq40,bq41_1,bq41_2,bq41_3"
    data[columns.split(",")] = data[columns.split(",")].replace(" ", 0).astype(int)

    columns = "bq12_2,bq12_3,bq12_4"
    data[columns.split(",")] = data[columns.split(",")].replace(" ", 9).astype(int)

    # Fill the missing values with `없음` word.
    columns = "bq5_2,bq19_1,bq30,bq31,bq32,bq33,bq34,bq38_1"
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
        "aq1_1,aq1_2,aq2_1,aq2_2,aq3_1,aq3_2,aq4_1,aq4_2,aq5_1,aq5_2,aq6_1,aq6_2,aq7_1,"
        "aq7_2,aq8_1,aq8_2,aq9_1,aq9_2,aq10_1,aq10_2,aq11_1,aq11_2,aq12_1,aq12_2,"
        "aq13_1,aq13_2,aq14_1,aq14_2,aq15_1,aq15_2,aq16_1,aq16_2,aq17_1,aq17_2,aq18_1,"
        "aq18_2,aq19_1,aq19_2,aq20_1,aq20_2,aq21_1,aq21_2,aq22_1,aq22_2,aq23_1,aq23_2,"
        "aq24_1,aq24_2,aq25_1,aq25_2,aq26_1,aq26_2,aq27_1,aq27_2,aq28_1,aq28_2,aq29_1,"
        "aq29_2,aq30_1,aq30_2,aq31_1,aq31_2,aq32_1,aq32_2,aq33_1,aq33_2,aq34_1,aq34_2,"
        "aq35_1,aq35_2,aq36_1,aq36_2,aq37_1,aq37_2,aq38_1,aq38_2,aq39_1,aq39_2,aq40_1,"
        "aq40_2,aq41_1,aq41_2,bq1,bq2,bq3,bq4,bq4_1_accum,bq5,bq5_1,bq5_2,bq6,bq7,"
        "bq8_1,bq8_2,bq8_3,bq9,bq10,bq11,bq12_1,bq12_2,bq12_3,bq12_4,bq12_5,bq13,bq14,"
        "bq15_1,bq15_2,bq15_3,bq16,bq17,bq18_1,bq18_2,bq18_3,bq18_4,bq18_5,bq18_6,"
        "bq18_7,bq19,bq19_1,bq20,bq21,bq22,bq23,bq24_1,bq24_2,bq24_3,bq24_4,bq24_5,"
        "bq24_6,bq24_7,bq24_8,bq25,bq26,bq27,bq28,bq29,bq30,bq31,bq32,bq33,bq34,bq35,"
        "bq36,bq38,bq38_1,bq39_1,bq39_2,bq40,bq41_1,bq41_2,bq41_3,knowcode"
    )
    return data.idx, data[[x for x in columns.split(",") if x in data.columns]]
