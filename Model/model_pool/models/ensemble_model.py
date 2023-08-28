import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
import torch.nn.init as init
import os

import warnings
warnings.filterwarnings("ignore")

class Ensemble_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.trained = False
        self.w = None
    
    def train(self, x_train, y_train):
        pass
    
    def forward(self, x_test: np.array):
        assert self.trained, "model should be trained first"
        return np.sum(self.w * x_test, axis=1, keepdims=True)
    
    def predict(self, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array, retrain_interval=None, max_retrain_samples=None, progress_bar=False):
        if retrain_interval == None: retrain_interval = x_test.shape[0]
        
        if max_retrain_samples == None: max_retrain_samples = x_train.shape[0]
        elif max_retrain_samples == -1: max_retrain_samples = x_train.shape[0] + x_test.shape[0]
        x = np.vstack((x_train, x_test))
        y = np.vstack((y_train, y_test))

        test_index = x_train.shape[0]
        
        preds = []
        iter = range(test_index, x.shape[0], retrain_interval)
        if progress_bar:
            from tqdm import tqdm
            iter = tqdm(iter)
        for i in iter:
            x_train_ = x[max(0, i-max_retrain_samples):i]
            y_train_ = y[max(0, i-max_retrain_samples):i]
            x_test_ = x[i:i+retrain_interval]
            self.train(x_train_, y_train_)
            preds.append(self.forward(x_test_))
        return np.vstack(preds)
        

class ReweightModel(Ensemble_model):
    def __init__(self, d_feat=5, hidden_size=64, num_layers=2, dropout=0.0, 
                 embedding_size=32, feature_size=32, sim_size=32, model_count=20, batch_size=800):
        super().__init__()
        self.model_count = model_count + 1
        self.batch_size = batch_size
        self.sim_size = sim_size

        self.reweighter = nn.Linear(sim_size, 1)
        self.bn = nn.BatchNorm1d(num_features=embedding_size, )
        self.trained = False
        
        self.d_feat = d_feat
        
    
    def normParam(self, data, weight):
        miu_w = weight.T @ data
        sig_w = (data-miu_w).T @ ((data-miu_w) * weight)
        return sig_w
    
    
    def reweight1(self, ydata):
        cols = ydata.shape[1]
        n = len(ydata) // 2 // self.sim_size * self.sim_size
        assert n > 0
        xx = ydata[len(ydata)-n:].reshape((self.sim_size, -1, cols)).mean(axis=1)
        reweight_feature = ((torch.unsqueeze(ydata, 1) - torch.unsqueeze(xx, 0)) ** 2).sum(axis=2)
        weight = F.softmax(self.reweighter(reweight_feature), dim=0)
        
        return weight

    
    def train(self, x_train, y_train):
        self.trained = True
        
        x_train, y_train = map(lambda x: torch.from_numpy(x).float(), (x_train, y_train))
        
        weight = self.reweight1(torch.hstack((y_train, x_train)))
        sig = self.normParam(y_train-x_train, weight)
        w = torch.sum(torch.linalg.pinv(sig), dim=0, keepdim=True)
        if torch.any(w < 0): w += w.min()
        self.w = (w / w.sum()).detach().numpy()
        
        
class PerfomanceBasedModel(ReweightModel):
    def __init__(self):
        super().__init__()
        
    def train(self, x_train: np.array, y_train: np.array):
        self.trained = True
        
        w = 1 / np.mean(np.abs(y_train - x_train), axis=0, keepdims=True)
        self.w = w / w.sum()