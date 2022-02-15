import pandas as pd


def clean_know_2020(data: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    data = data.copy()

    # Convert the data type to integer and fill the missing values.
    for i in range(1, 45):
        data[f"saq{i}_1"] = data[f"saq{i}_1"].astype(int)
        data[f"saq{i}_2"] = data[f"saq{i}_2"].replace(" ", 0).astype(int)

    # Convert the data type to integer.
    for i in range(1, 14):
        data[f"vq{i}"] = data[f"vq{i}"].astype(int)

    # Convert the data type to integer.
    columns = (
        "bq1,bq2,bq3,bq4,bq5,bq6,bq7,bq8_1,bq8_2,bq8_3,bq9,bq10,bq11,bq12_1,bq12_2,"
        "bq12_3,bq12_4,bq12_5,bq13_1,bq13_2,bq13_3,bq14_1,bq14_2,bq14_3,bq14_4,bq14_5,"
        "bq14_6,bq14_7,bq15,bq16_1,bq16_2,bq16_3,bq16_4,bq16_5,bq16_6,bq16_7,bq16_8,"
        "bq16_9,bq16_10,bq17,bq18_1,bq18_2,bq18_3,bq18_4,bq18_5,bq18_6,bq18_7,bq18_8,"
        "bq18_9,bq19,bq20,bq21_1,bq21_2,bq21_3,bq21_4,bq22_1,bq22_2,bq22_3,bq22_4,"
        "bq22_5,bq22_6,bq23_1,bq23_2,bq23_3,bq24,bq26"
    )
    data[columns.split(",")] = data[columns.split(",")].astype(int)

    # Fill the missing values and convert to integer.
    columns = "bq5_1,bq25,bq27_1,bq27_2,bq28,bq29,bq30_1,bq30_2,bq30_3"
    data[columns.split(",")] = data[columns.split(",")].replace(" ", 0).astype(int)

    # Fill the missing values with `없음` word.
    columns = "bq5_2,bq18_10,bq20_1,bq26_1"
    data[columns.split(",")] = data[columns.split(",")].fillna("없음")
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
        "saq1_1,saq1_2,saq2_1,saq2_2,saq3_1,saq3_2,saq4_1,saq4_2,saq5_1,saq5_2,saq6_1,"
        "saq6_2,saq7_1,saq7_2,saq8_1,saq8_2,saq9_1,saq9_2,saq10_1,saq10_2,saq11_1,"
        "saq11_2,saq12_1,saq12_2,saq13_1,saq13_2,saq14_1,saq14_2,saq15_1,saq15_2,"
        "saq16_1,saq16_2,saq17_1,saq17_2,saq18_1,saq18_2,saq19_1,saq19_2,saq20_1,"
        "saq20_2,saq21_1,saq21_2,saq22_1,saq22_2,saq23_1,saq23_2,saq24_1,saq24_2,"
        "saq25_1,saq25_2,saq26_1,saq26_2,saq27_1,saq27_2,saq28_1,saq28_2,saq29_1,"
        "saq29_2,saq30_1,saq30_2,saq31_1,saq31_2,saq32_1,saq32_2,saq33_1,saq33_2,"
        "saq34_1,saq34_2,saq35_1,saq35_2,saq36_1,saq36_2,saq37_1,saq37_2,saq38_1,"
        "saq38_2,saq39_1,saq39_2,saq40_1,saq40_2,saq41_1,saq41_2,saq42_1,saq42_2,"
        "saq43_1,saq43_2,saq44_1,saq44_2,vq1,vq2,vq3,vq4,vq5,vq6,vq7,vq8,vq9,vq10,vq11,"
        "vq12,vq13,bq1,bq2,bq3,bq4,bq4_1_accum,bq5,bq5_1,bq5_2,bq6,bq7,bq8_1,bq8_2,"
        "bq8_3,bq9,bq10,bq11,bq12_1,bq12_2,bq12_3,bq12_4,bq12_5,bq13_1,bq13_2,bq13_3,"
        "bq14_1,bq14_2,bq14_3,bq14_4,bq14_5,bq14_6,bq14_7,bq15,bq16_1,bq16_2,bq16_3,"
        "bq16_4,bq16_5,bq16_6,bq16_7,bq16_8,bq16_9,bq16_10,bq17,bq18_1,bq18_2,bq18_3,"
        "bq18_4,bq18_5,bq18_6,bq18_7,bq18_8,bq18_9,bq18_10,bq19,bq20,bq20_1,bq21_1,"
        "bq21_2,bq21_3,bq21_4,bq22_1,bq22_2,bq22_3,bq22_4,bq22_5,bq22_6,bq23_1,bq23_2,"
        "bq23_3,bq24,bq25,bq26,bq26_1,bq27_1,bq27_2,bq28,bq29,bq30_1,bq30_2,bq30_3,"
        "knowcode"
    )
    return data.idx, data[[x for x in columns.split(",") if x in data.columns]]
