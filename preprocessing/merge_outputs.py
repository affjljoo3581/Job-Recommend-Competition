import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="+")
    args = parser.parse_args()

    datasets = defaultdict(list)
    for filename in args.filename:
        match = re.match(r"(.*)-(simcse|bert).pkl", filename)
        if match:
            datasets[match.group(1)].append(filename)

    for dataset_name, dataset_filenames in datasets.items():
        dataset_dict = {"text_embeddings": [], "text_columns": []}

        for filename in sorted(dataset_filenames):
            with open(filename, "rb") as fp:
                current_dict = pickle.load(fp)
            text_embeddings = current_dict.pop("text_embeddings")
            text_columns = current_dict.pop("text_columns")

            dataset_dict.update(current_dict)
            dataset_dict["text_embeddings"].append(text_embeddings)
            dataset_dict["text_columns"] += text_columns

            os.remove(filename)

        dataset_dict["text_embeddings"] = np.concatenate(
            dataset_dict["text_embeddings"], axis=1
        )
        with open(f"{dataset_name}.pkl", "wb") as fp:
            pickle.dump(dataset_dict, fp)
