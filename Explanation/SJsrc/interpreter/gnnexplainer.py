import dgl
import torch
from math import sqrt


class GNNExplainer:
    def __init__(self, model, epochs=100, lr=0.01, device='cpu'):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.lr = lr
        # sparsity coeffs
        self.coeffs = {
            'pred_loss': 1,
            'edge_size': 1e-4,
            'edge_ent': 1e-2,
        }
        self.MIN = -1e9

    def dense2sparse(self, rel_matrix, feature, device):
        # convert adj matrix to dgl sparse graph
        if len(rel_matrix.shape) == 3:
            rel_matrix = rel_matrix.sum(axis=-1)  # [N, N]
        idx = rel_matrix.nonzero(as_tuple=True)
        dgl_graph = dgl.graph((idx[0], idx[1])).to(device)
        dgl_graph.ndata['nfeat'] = torch.Tensor(feature).to(device)
        return dgl_graph

    def sparse2dense(self, rel_matrix, g):
        g_adj = torch.zeros(g.num_nodes(), g.num_nodes(), rel_matrix.shape[-1])
        src, dst = g.edges()
        g_nids = g.ndata[dgl.NID]
        g_adj[src, dst, :] = rel_matrix[g_nids[src], g_nids[dst], :]
        return g_adj

    def set_mask(self, adj):
        N = adj.size(0)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn_like(adj) * std).to(self.device)
        # set the non-existing edges to -inf
        self.edge_mask.data[adj == 0] = self.MIN

    def loss(self, pred, orig_pred, EPS=1e-15):
        # pred = log_logits.softmax(dim=1)[0, pred_label]
        # loss = -torch.log2(pred+ EPS) + torch.log2(1 - pred+ EPS)

        l1 = (pred.item() - orig_pred) ** 2
        l1 = self.coeffs['pred_loss'] * l1
        m = self.edge_mask.sigmoid()
        l2 =  self.coeffs['edge_size'] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        l3 =  self.coeffs['edge_ent'] * ent.mean()
        # print(f'Loss: {l1.item():.4f}, {l2.item():.4f}, {l3.item():.4f}')
        return l1+l2+l3

    def explain(self, original_preds, g, rel_matrix, stock_id, top_k):
        target_id = stock_id
        g_c = dgl.khop_in_subgraph(g, target_id, 1)[0]

        origin_ids = g_c.ndata['_ID'].tolist()
        new_target_id = origin_ids.index(target_id)
        original_pred = original_preds[stock_id]

        g_c_adj = self.sparse2dense(rel_matrix, g_c)
        self.set_mask(g_c_adj)
        self.model.to(self.device)
        optimizer = torch.optim.Adam([self.edge_mask], lr=self.lr)
        for epoch in range(self.epochs):
            self.model.eval()
            optimizer.zero_grad()
            pred = self.model(g_c.ndata['nfeat'].to(self.device), self.edge_mask.sigmoid())[new_target_id]
            loss = self.loss(pred, original_pred)
            loss.backward()
            optimizer.step()
            self.edge_mask.data[g_c_adj == 0] = self.MIN
            # if epoch % 10 == 0:
            #     print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

        edge_mask = self.edge_mask.sigmoid().detach().cpu()
        # get the top k edges
        topk_v, topk_i = edge_mask.sum(2)[:,new_target_id].topk(top_k)
        explanation = {origin_ids[topk_i[i]]: [topk_v[i].item(), edge_mask[topk_i[i], new_target_id, :]] for i in range(top_k)}
        for k, v in explanation.items():
            l = v[1].nonzero().view(-1)
            explanation[k][1] = {i.item(): v[1][i].item() for i in l}
        return explanation



