import argparse
import os
import warnings

import pandas as pd
import torch
import tqdm

from dataset import load_test_dataset
from modeling import SiDClassifier, SiDConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


@torch.no_grad()
def predict(args: argparse.Namespace):
    state_dict = torch.load(args.model_name)
    config, state_dict = state_dict["config"], state_dict["state_dict"]

    model_kwargs, index, label_list, dataloader = load_test_dataset(
        args.dataset_filename, args.batch_size
    )
    model = SiDClassifier(SiDConfig(**model_kwargs, **config.model))
    model.load_state_dict(state_dict)
    model.eval().cuda()

    preds = []
    for batch in tqdm.tqdm(dataloader):
        preds += model(*batch).argmax(-1).tolist()

    # Save the prediction to a csv file.
    preds = [{"idx": i, "knowcode": label_list[p]} for i, p in zip(index, preds)]
    pd.DataFrame(preds).to_csv(args.model_name.replace(".pth", ".csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("dataset_filename")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    predict(args)
