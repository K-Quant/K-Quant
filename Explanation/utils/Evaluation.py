import numpy as np
from sklearn.metrics import ndcg_score

def metric_fn(preds):
    preds = preds[~np.isnan(preds['label'])]
    precision = {}
    recall = {}
    ndcg = {}
    temp = preds.groupby(level='datetime').apply(lambda x: x.sort_values(by='score', ascending=False))
    temp2 = preds.groupby(level='datetime').apply(lambda x: x.sort_values(by='label', ascending=False))
    if len(temp.index[0]) > 2:
        temp = temp.reset_index(level=0).drop('datetime', axis=1)
    if len(temp2.index[0]) > 2:
        temp2 = temp2.reset_index(level=0).drop('datetime', axis=1)

    for k in [1, 3, 5, 10, 20, 30, 50, 100]:
        precision[k] = (temp.groupby(level='datetime').apply(lambda x: (x.label[:k] > 0).sum() / k)
                        / temp2.groupby(level='datetime').apply(lambda x: (x.label[:k] > 0).sum() / k)).mean()

        recall[k] = temp.groupby(level='datetime').apply(lambda x: (x.label[:k] > 0).sum() / (x.label > 0).sum()).mean()
        ndcg[k] = preds.groupby(level='datetime').apply(lambda x: ndcg_score([np_relu(x.score)],
                                                                             [np_relu(x.label)], k=k)).mean()

    rank_ic_mean = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score, method='spearman')).mean()
    rank_ic_std = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score, method='spearman')).std()
    return rank_ic_mean, rank_ic_std


def np_relu(x):
    return x * (x > 0)
