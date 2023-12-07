import numpy as np
from sklearn.metrics import ndcg_score


def cal_reliability(preds):
    preds = preds[~np.isnan(preds['label'])]
    rank_ic_mean = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score, method='spearman')).mean()

    return rank_ic_mean


def cal_stability(preds):
    preds = preds[~np.isnan(preds['label'])]
    rank_ic_std = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score, method='spearman')).std()
    return rank_ic_std


def np_relu(x):
    return x * (x > 0)
