from pytorch_training import train_epoch, eval_model

from models import CreditsRNN
from dataset_preprocessing_utils import features
from utils import compute_emb_projections, read_pickle_file
from training_aux import EarlyStopping

import hydra
from omegaconf import DictConfig

import os
import torch
import wandb

@hydra.main(config_path=".", config_name='gru', version_base=None)
def train_rnn_model(
    cfg: DictConfig
):
    wandb.init(project="dl-alpha-demo", config=cfg, name=cfg['run_name'])
    
    uniques = read_pickle_file(cfg['uniques_emb_path'])
    embedding_projections = compute_emb_projections(uniques)
    
    model = CreditsRNN(
        features, embedding_projections, 
        rnn_type=cfg['model']['type'], 
        rnn_units=cfg['model']['units'],
        rnn_num_layers=cfg['model']['nlayers'],
        top_classifier_units=cfg['model']['top_classifier_units']
    )
    
    optimizer = torch.optim.Adam(
        lr=cfg['optimizer']['lr'], 
        params=model.parameters(),
        weight_decay=cfg['optimizer']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=cfg['scheduler']['base_lr'], 
        step_size_up=cfg['scheduler']['step_size_up'], 
        max_lr=cfg['scheduler']['max_lr'], 
        cycle_momentum=cfg['scheduler']['cycle_momentum'], 
        mode=cfg['scheduler']['mode']
    )
    
    es = EarlyStopping(
        patience=3, mode="max", verbose=True, 
        save_path=os.path.join(cfg['path_to_checkpoints'], "best_checkpoint.pt"), 
        metric_name="ROC-AUC", save_format="torch")
    
    dataset_train = sorted([os.path.join(cfg['train_buckets_path'], x) 
                            for x in os.listdir(cfg['train_buckets_path'])])
    dataset_val = sorted([os.path.join(cfg['val_buckets_path'], x) 
                          for x in os.listdir(cfg['val_buckets_path'])])
   

    for epoch in range(cfg.num_epochs):
        print(f"Starting epoch {epoch+1}")
        train_epoch(model, optimizer, scheduler, dataset_train, batch_size=cfg['train_batch_size'], 
                    shuffle=True, print_loss_every_n_batches=1000, device=cfg['device'])
        
        val_roc_auc = eval_model(model, dataset_val, 
                                batch_size=cfg['val_batch_size'], 
                                device=cfg['device'])
        es(val_roc_auc, model)
        
        if es.early_stop:
            print("Early stopping reached. Stop training...")
            break

        torch.save(
            model.state_dict(), 
            os.path.join(cfg['path_to_checkpoints'], f"epoch_{epoch+1}_val_{val_roc_auc:.3f}.pt")
        )
        
        train_roc_auc = eval_model(model, dataset_train, 
                                batch_size=cfg['val_batch_size'], 
                                device=cfg['device'])
        print(f"Epoch {epoch+1} completed. Train ROC-AUC: {train_roc_auc}, val ROC-AUC: {val_roc_auc}")
        wandb.log({"train_roc_auc": train_roc_auc, "val_roc_auc" : val_roc_auc})
    wandb.finish()
    
if __name__ == '__main__':
    train_rnn_model()
