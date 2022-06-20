from pytorch_training import inference

from models import CreditsRNN
from dataset_preprocessing_utils import features
from utils import compute_emb_projections, read_pickle_file


import hydra
from omegaconf import DictConfig

import os
import torch

@hydra.main(config_path=".", config_name="gru", version_base=None)
def predict_rnn_model(
    cfg: DictConfig
):
    
    uniques = read_pickle_file(cfg['uniques_emb_path'])
    embedding_projections = compute_emb_projections(uniques)
    
    model = CreditsRNN(
        features, embedding_projections, 
        rnn_type=cfg['model']['type'], 
        rnn_units=cfg['model']['units'],
        rnn_num_layers=cfg['model']['nlayers'],
        top_classifier_units=cfg['model']['top_classifier_units']
    )
    
    model.load_state_dict(
        torch.load(os.path.join(cfg['path_to_checkpoints'], "best_checkpoint.pt")))
    model.eval()
    
    dataset_test = sorted([os.path.join(cfg['test_buckets_path'], x) 
                           for x in os.listdir(cfg['test_buckets_path'])])
    
    test_preds = inference(model, dataset_test, batch_size=256, device=cfg['device'])
    test_preds.to_csv("rnn_submission.csv", index=None)
    
if __name__ == '__main__':
    predict_rnn_model()
