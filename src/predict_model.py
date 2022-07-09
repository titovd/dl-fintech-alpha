from itertools import accumulate
from pytorch_training import inference

from models import CreditsRNN
from utils import compute_emb_projections, read_pickle_file, get_model_by_name


import hydra
from omegaconf import DictConfig

import os
import torch
from tqdm.auto import tqdm

@hydra.main(config_path=".", config_name="gru", version_base=None)
def predict_rnn_model(
    cfg: DictConfig
):
    dataset_test = sorted([os.path.join(cfg['test_buckets_path'], x) 
                           for x in os.listdir(cfg['test_buckets_path'])])
    
    accumulated_preds = None
    for fold in tqdm(range(1, 4)):
        model = get_model_by_name(
            cfg['model']['name'], cfg['uniques_emb_path'], **cfg['model']['params'])
        
        model.load_state_dict(
            torch.load(os.path.join(cfg['path_to_checkpoints'], f"{fold}_fold_best_checkpoint.pt")))
        model.eval()
        test_preds = inference(model, dataset_test, batch_size=256, device=cfg['device'])
        if accumulate is None:
            accumulated_preds = test_preds
        else:
            accumulated_preds['score'] += test_preds['score']
    
    accumulated_preds.to_csv(os.path.join(cfg['path_to_checkpoints'], cfg['submission_filename']), index=None)
    
if __name__ == '__main__':
    predict_rnn_model()
