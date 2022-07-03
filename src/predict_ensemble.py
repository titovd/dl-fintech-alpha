from pytorch_training import inference

from models import CreditsRNN
from utils import compute_emb_projections, read_pickle_file, get_model_by_name


import hydra
from omegaconf import DictConfig

import os
import torch

@hydra.main(config_path=".", config_name="gru", version_base=None)
def predict_ensemble(
    cfg: DictConfig
):
    pass
    # @TODO predictions using stackings
    # predictions.to_csv(os.path.join(cfg['path_to_checkpoints'], 'submission.csv'), index=None)
    
if __name__ == '__main__':
    predict_ensemble()
