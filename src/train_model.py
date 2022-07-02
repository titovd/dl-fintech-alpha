from pytorch_training import train_epoch, eval_model

from utils import get_model_by_name, get_optimizer_by_name, get_scheduler_by_name
from training_aux import EarlyStopping

import hydra
from omegaconf import DictConfig

import numpy as np
import os
import torch
import wandb
import random

def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@hydra.main(config_path=".", config_name='gru', version_base=None)
def train_model(
    cfg: DictConfig
):
    wandb.init(project="dl-alpha-demo", config=cfg, name=cfg['run_name'])
    seed_all(cfg['seed'])

    model = get_model_by_name(
        cfg['model']['name'], cfg['uniques_emb_path'], **cfg['model']['params'])
    optimizer = get_optimizer_by_name(
        cfg['optimizer']['name'], model, **cfg['optimizer']['params'])
    scheduler = get_scheduler_by_name(
        cfg['scheduler']['name'], optimizer, **cfg['scheduler']['params'])

    es = EarlyStopping(
        patience=3, mode="max", verbose=True,
        save_path=os.path.join(
            cfg['path_to_checkpoints'], "best_checkpoint.pt"),
        metric_name="ROC-AUC", save_format="torch")

    dataset_train = sorted([os.path.join(cfg['train_buckets_path'], x)
                            for x in os.listdir(cfg['train_buckets_path'])])
    dataset_val = sorted([os.path.join(cfg['val_buckets_path'], x)
                          for x in os.listdir(cfg['val_buckets_path'])])

    for epoch in range(cfg.num_epochs):
        print(f"Starting epoch {epoch+1}")
        train_epoch(model, optimizer, scheduler, dataset_train, batch_size=cfg['train_batch_size'],
                    shuffle=True, print_loss_every_n_batches=100, device=cfg['device'])

        val_roc_auc = eval_model(model, dataset_val,
                                 batch_size=cfg['val_batch_size'],
                                 device=cfg['device'])
        es(val_roc_auc, model)

        if es.early_stop:
            print("Early stopping reached. Stop training...")
            break

        torch.save(
            model.state_dict(),
            os.path.join(cfg['path_to_checkpoints'],
                         f"epoch_{epoch+1}_val_{val_roc_auc:.3f}.pt")
        )

        train_roc_auc = eval_model(model, dataset_train,
                                   batch_size=cfg['val_batch_size'],
                                   device=cfg['device'])
        print(
            f"Epoch {epoch+1} completed. Train ROC-AUC: {train_roc_auc}, val ROC-AUC: {val_roc_auc}")
        wandb.log({"train_roc_auc": train_roc_auc, "val_roc_auc": val_roc_auc})
    wandb.finish()


if __name__ == '__main__':
    train_model()
