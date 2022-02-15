import sys

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer


@torch.no_grad()
def get_text_embeddings_from_simcse(
    package_dir: str, model_ckpt: str, texts: list[str], batch_size: int = 1024
) -> np.ndarray:
    """Calculate SimCSE text embeddings.

    Args:
        package_dir: The directory of simcse project.
        model_ckpt: The model checkpoint file of the simcse model.
        texts: The list of texts to be embedded.
        batch_size: The number of batch. Default is `1024`.

    Returns:
        A collection of embedded vectors.
    """
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)

    from data.dataloader import convert_to_tensor, example_model_setting

    model, transform, device = example_model_setting(model_ckpt)
    input_tensors = convert_to_tensor(texts, transform)

    input_tensor_keys = sorted(input_tensors.keys())
    input_tensor_values = (input_tensors[name] for name in input_tensor_keys)
    dataloader = DataLoader(TensorDataset(*input_tensor_values), batch_size)

    batch_text_embeddings = []
    for batch_input_tensors in tqdm.tqdm(dataloader):
        batch_input_tensors = dict(zip(input_tensor_keys, batch_input_tensors))
        batch_text_embeddings.append(model.encode(batch_input_tensors, device))

    text_embeddings = torch.cat(batch_text_embeddings, dim=0)
    return text_embeddings.cpu().numpy()


@torch.no_grad()
def get_text_embeddings_from_input_embeddings(
    model_name: str, texts: list[str], batch_size: int = 1024
) -> np.ndarray:
    """Calculate Transformer input embeddings.

    Args:
        model_name: The transformer model name.
        texts: The list of texts to be embedded.
        batch_size: The number of batch. Default is `1024`.

    Returns:
        A collection of embedded vectors.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embeddings = AutoModel.from_pretrained(model_name).get_input_embeddings()

    text_embeddings = []
    for batch_offset in tqdm.trange(0, len(texts), batch_size):
        encodings = tokenizer(
            texts[batch_offset : batch_offset + batch_size],
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

        # Average the input embedding vectors with excluding the mask tokens.
        masks = encodings["attention_mask"][:, :, None]
        embeds = embeddings(encodings["input_ids"])
        text_embeddings.append(((embeds * masks).sum(1) / masks.sum(1)).numpy())

    return np.concatenate(text_embeddings, axis=0)
