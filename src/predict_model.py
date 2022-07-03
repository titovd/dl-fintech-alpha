from pytorch_training import inference

from models import CreditsRNN
from utils import compute_emb_projections, read_pickle_file, get_model_by_name


import hydra
from omegaconf import DictConfig

import os
import torch

@hydra.main(config_path=".", config_name="gru", version_base=None)
def predict_rnn_model(
    cfg: DictConfig
):
    model = get_model_by_name(
        cfg['model']['name'], cfg['uniques_emb_path'], **cfg['model']['params'])
    
    model.load_state_dict(
        torch.load(os.path.join(cfg['path_to_checkpoints'], "best_checkpoint.pt")))
    model.eval()
    
    dataset_test = sorted([os.path.join(cfg['test_buckets_path'], x) 
                           for x in os.listdir(cfg['test_buckets_path'])])
    
    test_preds = inference(model, dataset_test, batch_size=256, device=cfg['device'])
    test_preds.to_csv(os.path.join(cfg['path_to_checkpoints'], 'submission.csv'), index=None)
    
if __name__ == '__main__':
    predict_rnn_model()
