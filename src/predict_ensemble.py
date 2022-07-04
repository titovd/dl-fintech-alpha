import hydra
from omegaconf import DictConfig

import pandas as pd
import pickle
import os
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def build_dataset(dataset_path, submission_filename, cfg):
    dataset = pd.read_csv(dataset_path)
    for i, model_submission_path in enumerate(cfg['submissions_paths']):
        model_submission = pd.read_csv(
            os.path.join(model_submission_path, submission_filename)
        )
        dataset = dataset.join(
            model_submission.set_index('id'), 
            on='id', how='inner', 
            rsuffix=f'_model_{i}'
        )
    return dataset

@hydra.main(config_path="/content/dl-fintech-alpha/config/", config_name="ensemble", version_base=None)
def predict_ensemble(
    cfg: DictConfig
):

    train_dataset = build_dataset(cfg['train_target_path'], "train_submission.csv", cfg)
    val_dataset = build_dataset(cfg['train_target_path'], "val_submission.csv", cfg)
    test_dataset = build_dataset(cfg['test_target_path'], "test_submission.csv", cfg)
    
    lr = LogisticRegression(penalty='none', solver='saga')
    lr.fit(train_dataset.drop(['flag', 'id'], axis=1), train_dataset['flag'])

    val_predict = lr.predict_proba(val_dataset.drop(['flag', 'id'], axis=1))
    print(f'Val ROC_AUC: {roc_auc_score(val_dataset["flag"], val_predict[:, 1])}')

    preds = lr.predict_proba(test_dataset.drop(['id'], axis=1))[:, 1]
    test_preds = pd.DataFrame({
        "id": test_dataset['id'].values,
        "score": preds
    })
    test_preds.to_csv("ensemble_submission_path.csv", index=None)
    
if __name__ == '__main__':
    predict_ensemble()