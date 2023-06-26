import torch
import numpy as np
import pandas as pd
from torchmetrics import RetrievalNormalizedDCG


def approxNDCGLoss(y_pred, y_true, eps=1, alpha=1.):
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
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=0, index=indices_pred)
    # true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    # just like relu, let all values that below 0 equal to 0
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[0] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum((torch.pow(2, y_true_sorted) - 1) / D, dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    # Here we approximate the ranking positions according to Eqs 19-20 and later approximate NDCG (Eq 21)
    scores_diffs = y_pred_sorted[:, None].repeat(1, y_pred_sorted.shape[0]) - \
                   y_pred_sorted[None, :].repeat(y_pred_sorted.shape[0], 1)
    # scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :])
    # when sigmoid = 0.5, then the diff = 0, which is the original value of itself
    sig = torch.sigmoid(-alpha * scores_diffs)
    approx_pos = 1. + torch.sum(sig * (sig > 0.5), dim=-1)
    approx_D = torch.log2(1. + approx_pos)
    approx_NDCG = torch.sum((G / approx_D), dim=-1)
    return -torch.mean(approx_NDCG)


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
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=0, index=indices_pred)
    # true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    # just like relu, let all values that below 0 equal to 0
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

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


# indexes = torch.tensor([0, 0, 0, 0, 0, 0, 0])
# preds = torch.tensor([2.1, 5.5, 6, 1.5, 2.1, 3.2, 4])
# # target = torch.tensor([False, False, True, False, True, False, True])
# target = torch.tensor([2.0, 5.0, 6.0, 1.0, 2, 3, 4])


indexes = torch.tensor([0, 0, 0, 0, 0, 0])
preds = torch.tensor([1, 2.0, 3, 4, 5, 4])
target = torch.tensor([1, 2.0, 3, 4, 5, 2])
ndcg = RetrievalNormalizedDCG(k=4)
print(ndcg(preds, target, indexes=indexes))
print(approxNDCGLoss(preds, target))
print(approxNDCGLoss_cutk(preds, target, k=4))
