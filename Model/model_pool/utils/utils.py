import torch
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from torchmetrics import RetrievalNormalizedDCG


def mse(pred, label):
    loss = (pred - label) ** 2
    return torch.mean(loss)


def mae(pred, label):
    loss = (pred - label).abs()
    return torch.mean(loss)


def cal_cos_similarity(x, y):  # the 2nd dimension of x and y are the same
    xy = x.mm(torch.t(y))
    x_norm = torch.sqrt(torch.sum(x * x, dim=1)).reshape(-1, 1)
    y_norm = torch.sqrt(torch.sum(y * y, dim=1)).reshape(-1, 1)
    cos_similarity = xy / x_norm.mm(torch.t(y_norm))
    cos_similarity[cos_similarity != cos_similarity] = 0
    return cos_similarity


def cal_convariance(x, y):  # the 2nd dimension of x and y are the same
    e_x = torch.mean(x, dim=1).reshape(-1, 1)
    e_y = torch.mean(y, dim=1).reshape(-1, 1)
    e_x_e_y = e_x.mm(torch.t(e_y))
    x_extend = x.reshape(x.shape[0], 1, x.shape[1]).repeat(1, y.shape[0], 1)
    y_extend = y.reshape(1, y.shape[0], y.shape[1]).repeat(x.shape[0], 1, 1)
    e_xy = torch.mean(x_extend * y_extend, dim=2)
    return e_xy - e_x_e_y


def np_relu(x):
    return x * (x > 0)


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

    ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score)).mean()
    rank_ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score, method='spearman')).mean()
    return precision, recall, ic, rank_ic, ndcg


def loss_ic(pred, label):
    """
    directly use 1-ic as the loss function
    """
    mask = ~torch.isnan(label)
    pred = pred[mask]
    label = label[mask]
    res = torch.stack([pred, label])
    # label = np.array(label[mask].cpu())
    return 1 - torch.corrcoef(res)[0, 1]


def pair_wise_loss(pred, label, alpha=0.05):
    """
    original loss function in RSR
    """
    mask = ~torch.isnan(label)
    pred = pred[mask]
    label = label[mask]
    pred_p = pred.unsqueeze(0)
    label_p = label.unsqueeze(0)
    all_one = torch.ones(mask.shape[0], 1, device=pred.device)
    pred_diff = torch.matmul(all_one, pred_p) - torch.matmul(pred_p.T, all_one.T)
    label_diff = torch.matmul(all_one, label_p) - torch.matmul(label_p.T, all_one.T)
    pair_wise = torch.mean(torch.nn.ReLU()(-pred_diff * label_diff))
    point_wise = mse(pred[mask], label[mask])
    return point_wise + alpha * pair_wise


def NDCG_loss(pred, label, alpha=0.05, k=100):
    """
    NDCG loss function
    """
    mask = ~torch.isnan(label)
    pred = pred[mask]
    label = label[mask]
    index = torch.zeros(label.shape, dtype=torch.int64, device=pred.device)
    ndcg = RetrievalNormalizedDCG(k=k)
    ndcg_loss = alpha * ndcg(pred, label, indexes=index)
    point_wise = mse(pred, label)
    # point-wise decrease, model better
    # ndcg increase, model better
    return point_wise - ndcg_loss


def NDCG_evaluation(preds):
    return preds.groupby(level='datetime').apply(lambda x: ndcg_score([x.ground_truth], [x.pred])).mean()


def approxNDCGLoss_cutk(y_pred, y_true, eps=1, alpha=1., k=20):
    """
    Loss based on approximate NDCG introduced in "A General Approximation Framework for Direct Optimization of
    Information Retrieval Measures". Please note that this method does not implement any kind of truncation.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param alpha: score difference weight used in the sigmoid function
    :return: loss value, a torch.Tensor
    """
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()
    # padded_mask = y_true == padded_value_indicator
    # y_pred[padded_mask] = float("-inf")
    # y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    # print(y_pred_sorted.grad_fn)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    # 按照pred的顺序来排列
    true_sorted_by_preds = torch.gather(y_true, dim=0, index=indices_pred)
    # true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    # just like relu, let all values that below 0 equal to 0
    true_sorted_by_preds.clamp_(min=0.0000)
    y_true_sorted.clamp_(min=0.)
    if k == -1:
        # not perform cut
        true_sort_by_preds_k = true_sorted_by_preds
        y_pred_sorted_k = y_pred_sorted
        y_true_sorted_k = y_true_sorted
    else:
        true_sort_by_preds_k = true_sorted_by_preds[:k]
        y_pred_sorted_k = y_pred_sorted[:k]
        y_true_sorted_k = y_true_sorted[:k]

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_true_sorted_k.shape[0] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum((torch.pow(2, y_true_sorted_k) - 1) / D, dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sort_by_preds_k) - 1) / maxDCGs[:, None]

    # Here we approximate the ranking positions according to Eqs 19-20 and later approximate NDCG (Eq 21)
    scores_diffs = y_pred_sorted_k[:, None].repeat(1, y_pred_sorted_k.shape[0]) - \
                   y_pred_sorted_k[None, :].repeat(y_pred_sorted_k.shape[0], 1)
    # scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :])
    # when sigmoid = 0.5, then the diff = 0, which is the original value of itself
    sig = torch.sigmoid(-alpha * scores_diffs)
    approx_pos = 1. + torch.sum(sig * (sig > 0.5), dim=-1)
    approx_D = torch.log2(1. + approx_pos)
    approx_NDCG = torch.sum((G / approx_D), dim=-1)
    return -torch.mean(approx_NDCG)


def ApproxNDCG_loss(pred, label, alpha=0.05, k=100):
    mask = ~torch.isnan(label)
    pred = pred[mask]
    label = label[mask]
    ndcg_part = approxNDCGLoss_cutk(pred, label, k=k) * alpha
    point_wise = mse(pred, label)
    return point_wise + ndcg_part


def softclass_NDCG(pred, label):
    """
    issues: if we use NDCG, get weight*prob, then most will drop into middle part, this is not helpful
            for example, with [3,2,1,0] related weights, we have two prob [0,0.5,0.5,0] and [0.5,0,0,0.5]
            those two have THE SAME weights, this is not RIGHT!
    :param pred: the output of model [B,N]
    :param label: the ground truth [B]
    :return:
    """
    m = torch.nn.Softmax(dim=1)
    pred_n = m(pred)  # [B,N]
    weights = [w*w for w in range(pred_n.shape[1])]
    default_weight = torch.Tensor(weights, device=pred_n.device)
    x = pred_n@default_weight
    group = pred.shape[1]
    mc_label = torch.zeros(label.shape[0], device=pred.device, dtype=torch.long)  # shape [B]
    for i in range(1, group):
        indices = torch.topk(label, int(label.shape[0]*i/group)).indices.to(device=pred.device)
        mc_label[indices] += 1
    mc_label = torch.square(mc_label)
    return x, mc_label


def cross_entropy(pred, label):
    """
    :param pred: the prediction result from model shape [B,N]
    :param label: the label from dataset is the score, we will divide those score into 4 groups as their label
    :return:
    """
    ce_loss = torch.nn.CrossEntropyLoss()
    group = pred.shape[1]
    mc_label = torch.zeros(label.shape[0], device=pred.device, dtype=torch.long)  # shape [B]
    for i in range(1, group):
        indices = torch.topk(label, int(label.shape[0]*i/group)).indices.to(device=pred.device)
        mc_label[indices] += 1
    return ce_loss(pred, mc_label)


def class_approxNDCG(pred, label):
    """
    :param pred: the prediction result from model shape [B,N]
    :param label: the label from dataset is the score, we will divide those score into 4 groups as their label
    we do softmax on pred, then times the [3,2,1,0] matrix to get the final weights, the label weights are from
    0 to 3, and compute the approxNDCG
    """
    soft_x, mc_label = softclass_NDCG(pred, label)  # shape [B]
    return approxNDCGLoss_cutk(soft_x, mc_label.float(), k=-1)


def generate_label(pred, label):
    group = pred.shape[1]
    mc_label = torch.zeros(label.shape[0], device=pred.device, dtype=torch.long)  # shape [B]
    for i in range(1, group):
        indices = torch.topk(label, int(label.shape[0]*i/group)).indices
        mc_label[indices] += 1  # shape B
    pred_label = torch.topk(pred, 1).indices.squeeze()  # shape B
    return pred_label, mc_label


def evaluate_mc(preds):
    # default precision setting is micro
    acc = preds.groupby(level='datetime').apply(lambda x: accuracy_score(x['ground_truth'],x['pred'])).mean()
    average_precision = preds.groupby(level='datetime')\
        .apply(lambda x: precision_score(x['ground_truth'], x['pred'], average='micro')).mean()
    f1_micro = preds.groupby(level='datetime').\
        apply(lambda x: f1_score(x['ground_truth'], x['pred'], average='micro')).mean()
    f1_macro = preds.groupby(level='datetime').\
        apply(lambda x: f1_score(x['ground_truth'], x['pred'], average='macro')).mean()
    # roc_auc = preds.groupby(level='datetime').\
    #     apply(lambda x: roc_auc_score(x['ground_truth'], x['pred_array'], average='macro')).mean()

    return acc, average_precision, f1_micro, f1_macro


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value







