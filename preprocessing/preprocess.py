import argparse
import os
import pickle

import numpy as np
import pandas as pd

from cleaning import clean_know_data
from encoding import (
    encode_categorical_data_with_vocabulary,
    encode_numerical_data_to_gauss_rank,
    encode_numerical_data_to_standard_normalization,
)
from text_embedding import (
    get_text_embeddings_from_input_embeddings,
    get_text_embeddings_from_simcse,
)


def main(args: argparse.Namespace):
    train_idx, train_data = clean_know_data(
        pd.read_csv(args.input_train_file), args.data_type
    )
    test_idx, test_data = clean_know_data(
        pd.read_csv(args.input_test_file), args.data_type
    )

    # Extract the text, categorical and numerical features from train dataset.
    text_features = train_data.columns[train_data.dtypes == object].tolist()
    categorical_features = train_data.columns[
        (train_data.nunique() <= args.max_categories) & (train_data.dtypes != object)
    ].tolist()

    if args.label_name not in categorical_features:
        categorical_features += [args.label_name]

    numerical_features = sorted(
        set(train_data.columns) - set(text_features) - set(categorical_features)
    )

    # Collect the numerical features by normalizing the original data with rank-gauss
    # and standardization
    train_numerical_data, test_numerical_data = [], []

    for name in numerical_features:
        train_vals, test_vals, _ = encode_numerical_data_to_gauss_rank(
            train_data[name].to_numpy(),
            test_data[name].to_numpy(),
            epsilon=1e-4,
            noise_scale=0.001,
        )
        train_numerical_data.append(train_vals)
        test_numerical_data.append(test_vals)

    for name in numerical_features:
        train_vals, test_vals, _ = encode_numerical_data_to_standard_normalization(
            train_data[name].to_numpy(), test_data[name].to_numpy()
        )
        train_numerical_data.append(train_vals)
        test_numerical_data.append(test_vals)

    # Collect the categorical features.
    train_labels, label_list = None, None
    train_categorical_data, test_categorical_data, categories_list = [], [], []
    for name in categorical_features:
        train_cat_idx, test_cat_idx, vocab = encode_categorical_data_with_vocabulary(
            train_data[name].to_numpy(),
            test_data[name] if name != args.label_name else None,
            unknown_item_idx=0,
        )
        if name == args.label_name:
            train_labels, label_list = train_cat_idx, vocab
        else:
            train_categorical_data.append(train_cat_idx)
            test_categorical_data.append(test_cat_idx)
            categories_list.append(vocab)

    categorical_features = [x for x in categorical_features if x != args.label_name]

    # Stack the numerical and categorical data to the single arrays.
    train_numerical_data = np.stack(train_numerical_data, axis=1).astype(np.float32)
    train_categorical_data = np.stack(train_categorical_data, axis=1).astype(np.int64)
    test_numerical_data = np.stack(test_numerical_data, axis=1).astype(np.float32)
    test_categorical_data = np.stack(test_categorical_data, axis=1).astype(np.int64)

    # Embed the text features using SimCSE or Transformer input embeddings.
    train_text_data = [train_data[name].tolist() for name in text_features]
    test_text_data = [test_data[name].tolist() for name in text_features]

    num_text_embeddings, train_text_embeddings, test_text_embeddings = 0, [], []

    if args.use_simcse_embeddings:
        simcse_embeddings = get_text_embeddings_from_simcse(
            args.simcse_package_dir,
            args.simcse_model_ckpt,
            sum(train_text_data + test_text_data, []),
            args.embedding_batch_size,
        )
        # Since we stacked all texts (including train and test texts) to the single list
        # to reduce the duplicated preparation time, we need to split the stacked output
        # embedding arrays.
        train_simcse_text_embeddings = simcse_embeddings[
            : len(train_text_data) * len(train_text_data[0])
        ].reshape(len(train_text_data), len(train_text_data[0]), -1)
        test_simcse_text_embeddings = simcse_embeddings[
            len(train_text_data) * len(train_text_data[0]) :
        ].reshape(len(test_text_data), len(test_text_data[0]), -1)

        num_text_embeddings += 1
        train_text_embeddings.append(train_simcse_text_embeddings)
        test_text_embeddings.append(test_simcse_text_embeddings)

    if args.use_bert_embeddings:
        for bert_model_name in args.bert_model_name:
            bert_embeddings = get_text_embeddings_from_input_embeddings(
                bert_model_name, sum(train_text_data + test_text_data, [])
            )
            # Since we stacked all texts (including train and test texts) to the single
            # list to reduce the duplicated preparation time, we need to split the
            # stacked output embedding arrays.
            train_bert_text_embeddings = bert_embeddings[
                : len(train_text_data) * len(train_text_data[0])
            ].reshape(len(train_text_data), len(train_text_data[0]), -1)
            test_bert_text_embeddings = bert_embeddings[
                len(train_text_data) * len(train_text_data[0]) :
            ].reshape(len(test_text_data), len(test_text_data[0]), -1)

            num_text_embeddings += 1
            train_text_embeddings.append(train_bert_text_embeddings)
            test_text_embeddings.append(test_bert_text_embeddings)

    # Stack the text embeddings to the single arrays.
    train_text_embeddings = np.concatenate(train_text_embeddings, axis=0)
    train_text_embeddings = train_text_embeddings.transpose(1, 0, 2).astype(np.float32)

    test_text_embeddings = np.concatenate(test_text_embeddings, axis=0)
    test_text_embeddings = test_text_embeddings.transpose(1, 0, 2).astype(np.float32)

    # Create dataset info tables and save them to the output files.
    train_dataset_dict = {
        "index": train_idx.tolist(),
        "numerical": train_numerical_data,
        "categorical": train_categorical_data,
        "text_embeddings": train_text_embeddings,
        "labels": train_labels,
        "numerical_columns": numerical_features * 2,
        "categorical_columns": categorical_features,
        "text_columns": text_features * num_text_embeddings,
        "categories_list": categories_list,
        "label_list": label_list,
    }
    test_dataset_dict = {
        "index": test_idx.tolist(),
        "numerical": test_numerical_data,
        "categorical": test_categorical_data,
        "text_embeddings": test_text_embeddings,
        "numerical_columns": numerical_features * 2,
        "categorical_columns": categorical_features,
        "text_columns": text_features * num_text_embeddings,
        "categories_list": categories_list,
        "label_list": label_list,
    }

    postfix = []
    if args.use_simcse_embeddings:
        postfix.append("simcse")
    if args.use_bert_embeddings:
        postfix.append("bert")

    output_train_file = os.path.basename(args.input_train_file)
    output_train_file = output_train_file.replace(".csv", f"-{'-'.join(postfix)}.pkl")

    output_test_file = os.path.basename(args.input_test_file)
    output_test_file = output_test_file.replace(".csv", f"-{'-'.join(postfix)}.pkl")

    with open(output_train_file, "wb") as fp:
        pickle.dump(train_dataset_dict, fp)
    with open(output_test_file, "wb") as fp:
        pickle.dump(test_dataset_dict, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_train_file")
    parser.add_argument("input_test_file")
    parser.add_argument("--data_type", type=int, default=2017)
    parser.add_argument("--max_categories", type=int, default=25)
    parser.add_argument("--label_name", default="knowcode")
    parser.add_argument("--use_simcse_embeddings", action="store_true", default=False)
    parser.add_argument("--simcse_package_dir", default="./KoSimCSE")
    parser.add_argument("--simcse_model_ckpt", default="./KoSimCSE/nli_checkpoint.pt")
    parser.add_argument("--use_bert_embeddings", action="store_true", default=False)
    parser.add_argument(
        "--bert_model_name",
        nargs="*",
        default=["monologg/koelectra-base-v3-discriminator", "hyunwoongko/kobart"],
    )
    parser.add_argument("--embedding_batch_size", type=int, default=2048)
    main(parser.parse_args())
