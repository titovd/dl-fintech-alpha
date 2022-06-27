import os
from pickletools import optimize
import numpy as np
import pandas as pd
import pickle
import tqdm
import torch
from typing import List

from models import CreditsRNN, TransformerCreditsModel
from dataset_preprocessing_utils import features


def read_parquet_dataset_from_local(path_to_dataset: str, start_from: int = 0, num_parts_to_read: int = 2,
                                    columns: List[str] = None, verbose: bool = False) -> pd.DataFrame:
    """
    Читает ``num_parts_to_read партиций`` и преобразует их к pandas.DataFrame.

    Параметры:
    -----------
    path_to_dataset: str
        Путь до директории с партициями.
    start_from: int, default=0
        Номер партиции, с которой начать чтение.
    num_parts_to_read: int, default=2
        Число партиций, которые требуется прочитать.
    columns: List[str], default=None
        Список колонок, которые нужно прочитать из каждой партиции. Если None, то считываются все колонки.

    Возвращаемое значение:
    ----------------------
    frame: pandas.DataFrame
        Прочитанные партиции, преобразованные к pandas.DataFrame.
    """

    res = []
    start_from = max(0, start_from)
    # dictionory of format {partition number: partition filename}
    dataset_paths = {int(os.path.splitext(filename)[0].split("_")[-1]): os.path.join(path_to_dataset, filename)
                     for filename in os.listdir(path_to_dataset)}
    chunks = [dataset_paths[num] for num in sorted(
        dataset_paths.keys()) if num >= start_from][:num_parts_to_read]

    if verbose:
        print("Reading chunks:", *chunks, sep="\n")
    for chunk_path in tqdm.tqdm_notebook(chunks, desc="Reading dataset with pandas"):
        chunk = pd.read_parquet(chunk_path, columns=columns)
        res.append(chunk)
    return pd.concat(res).reset_index(drop=True)


def read_pickle_file(path: str):
    """ Загрузка файла через pickle по имени path. """
    with open(path, 'rb') as f:
        file = pickle.load(f)
        return file


def compute_embed_dim(n_cat: int) -> int:
    """ Вычисление размера эмбеддинга по количеству значений кат. фич. """
    return min(600, round(1.6 * n_cat**0.56))


def compute_emb_projections(uniques_features: dict) -> dict:
    return {feat: (max(uniq)+1, compute_embed_dim(max(uniq)+1)) for feat, uniq in uniques_features.items()}


def calculate_weight(ids: np.array) -> torch.FloatTensor:
    return torch.FloatTensor((ids / 2_500_000) * 0.55 + 0.85)


def get_model_by_name(model_name: str, emb_path: str, **kwargs) -> torch.nn.Module:
    uniques = read_pickle_file(emb_path)
    embedding_projections = compute_emb_projections(uniques)

    if model_name == "RNN":
        return CreditsRNN(
            features, embedding_projections, **kwargs)
    elif model_name == "Transformer":
        return TransformerCreditsModel(features, embedding_projections)


def get_optimizer_by_name(optimizer_name: str, model: torch.nn.Module, **kwargs) -> torch.optim.Optimizer:
    opt_dict = {
        'Adam': torch.optim.Adam,
    }

    assert optimizer_name in opt_dict.keys()
    return opt_dict[optimizer_name](params=model.parameters(), **kwargs)


def get_scheduler_by_name(scheduler_name: str, optimizer, **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
    scheduler_dict = {
        'StepLR': torch.optim.lr_scheduler.StepLR,
        'CyclicLR': torch.optim.lr_scheduler.CyclicLR
    }

    assert scheduler_name in scheduler_dict.keys()
    return scheduler_dict[scheduler_name](optimizer, **kwargs)
