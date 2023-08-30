import gc

import torch.nn as nn
import torch


class GradExplainer(nn.Module):
    def __init__(
            self,
            model
    ):
        super(GradExplainer, self).__init__()
        self.model = model

    def run_explain(self, feat, adj):
        self.model.zero_grad()
        adj.requires_grad = True
        if adj.grad is not None:
            adj.grad.zero_()
        ypred = self.model(feat, adj)
        loss = ypred.sum()
        loss.backward()
        masked_adj = adj.grad
        edge_weight_matrix = torch.sum(masked_adj, 2) / masked_adj.shape[2]
        return edge_weight_matrix










