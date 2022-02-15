import argparse
import re
from collections import defaultdict

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+")
    parser.add_argument("--merge", action="store_true", default=False)
    parser.add_argument("--ensemble", action="store_true", default=False)
    args = parser.parse_args()

    predictions = defaultdict(dict)
    for filename in args.predictions:
        pred = pd.read_csv(filename, index_col="idx")
        year = re.match(r".*(\d\d\d\d).*", filename).group(1)
        rs = re.match(r".*(rs\d+).*", filename).group(1)
        predictions[year][rs] = pred

    if args.ensemble:
        for year, preds in predictions.items():
            preds = pd.concat(preds.values(), axis=1).mode(axis=1)[[0]]
            preds = preds.rename(columns={0: "knowcode"}).astype(int)
            predictions[year] = {"ensemble": preds}

    if args.merge:
        for name in predictions[list(predictions.keys())[0]]:
            pd.concat(
                preds[name] for preds in predictions.values()
            ).sort_index().to_csv(f"submission-{name}.csv")
    else:
        for year, preds in predictions.items():
            for name, pred in preds.items():
                pred.to_csv(f"submission-{year}-{name}.csv")
