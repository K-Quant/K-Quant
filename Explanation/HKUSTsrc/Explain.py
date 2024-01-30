import argparse
import math
import pickle
import datetime
import pandas as pd
import random

import torch
from tqdm import tqdm

# from Config import config
from Explanation.HKUSTsrc.Explainer import *
from Explanation.HKUSTsrc.model import *
from Explanation.SJsrc.interpreter import xpath
from Explanation.utils.Evaluation import metric_fn
from Model.model_pool.models.model import relation_GATs


# from dataloader import DataLoader, create_loaders


class Explanation:
    def __init__(self, args, data_loader, explainer_name='InputGradientExplainer'):
        self.args = args
        # device
        self.device = args.device
        # data
        self.data = None
        self.graph_data = None
        self.data_loader = None
        self.graph_data_path = args.graph_data_path
        # model
        self.d_feat = args.d_feat
        self.num_layers = args.num_layers
        self.model_dir = args.model_dir
        self.num_relation = None
        self.pred_model = None

        # explainer
        self.explainer_name = explainer_name
        self.explainer = None
        self.EG = None

        #
        self.explained_graph_list = []
        self.explained_graph_dict = {explainer_name: []}
        self.evaluation_results = []
        self.eval_results_df = None
        # Explanation preparation
        self.load_data()
        self.get_pred_model()
        self.get_explainer()
        self.get_data_loader(data_loader)

    def get_pred_model(self):
        with torch.no_grad():
            if self.args.graph_model == 'NRSR':
                self.pred_model = NRSR(num_relation=self.num_relation,
                                       d_feat=self.d_feat,
                                       num_layers=self.num_layers)
            elif self.args.graph_model == 'relation_GATs':
                self.pred_model = relation_GATs(d_feat=self.d_feat, num_layers=self.num_layers)

            self.pred_model.to(self.device)
            self.pred_model.load_state_dict(torch.load(self.model_dir + '/model.bin', map_location=self.device))

    def get_explainer(self):
        if self.explainer_name == 'gnnExplainer':
            self.explainer = GNNExplainer(self.pred_model, self.args)

        elif self.explainer_name == 'inputGradientExplainer':
            self.explainer = InputGradientExplainer(self.pred_model)

        elif self.explainer_name == 'gradExplainer':
            self.explainer = GradExplainer(self.pred_model)

        elif self.explainer_name == 'effectExplainer':
            self.explainer = EffectExplainer(self.pred_model)

        elif self.explainer_name == 'random':
            pass

        elif self.explainer_name == 'xpathExplainer':
            self.explainer = xpath.xPath(num_layers=1, device=self.device)

    def load_data(self):
        self.graph_data = torch.Tensor(np.load(self.graph_data_path)).to(self.device)
        self.num_relation = self.graph_data.shape[2]

    def get_data_loader(self, data_loader):
        self.data_loader = data_loader

    def cal_reliability_stability(self):
        data_loader = self.data_loader
        preds = []
        for i, slc in tqdm(self.data_loader.iter_daily(), total=self.data_loader.daily_length):
            feature, label, market_value, stock_index, index = data_loader.get(slc)
            graph = self.graph_data[stock_index][:, stock_index]
            pred = self.pred_model(feature, graph)
            preds.append(pd.DataFrame({'score': pred.detach().numpy(), 'label': label.detach().numpy(), }, index=index))
        preds = pd.concat(preds, axis=0)
        reliability, stability = metric_fn(preds)
        return reliability, stability

    def explain(self):
        data_loader = self.data_loader
        exp_result_dict = {}
        for i, slc in tqdm(self.data_loader.iter_daily(), total=self.data_loader.daily_length):
            feature, label, market_value, stock_index, index = data_loader.get(slc)
            date = datetime.datetime.date(index[0][0])
            graph = self.graph_data[stock_index][:, stock_index]
            expl_graph = self.explainer.run_explain(feature, graph)
            exp_result_dict[str(date)] = {'expl_graph': expl_graph.detach().numpy(),
                                          'origin_graph': graph.detach().numpy(),
                                          'feature': feature.detach().numpy(),
                                          'stock_index_in_adj': stock_index.detach().numpy()}
        return exp_result_dict

    def explain_xpath(self, stock_list=None, get_fidelity=False, top_k=5, relation_list=None):
        # xpathExplainer params:
        #   stock_list->list of stock name, if None, explain all stocks
        #   get_fidelity->bool, if True, return fidelity in the result dict
        #  top_k->int, top k related stocks to be returned
        # An example of exp_result_dict:
        #  {'2019-01-02': {'SH600000': {'SH600015': {'score':0.22}, 'SH601166': {'score': 0.07}, ...}}}
        #if relation_list is not None, then exp_result_dict:
        #  {'2019-01-02': {'SH600000': {'SH600015': {'score':0.22, 'relations': [...]}, 'SH601166': {'score': 0.07, 'relations': [...]}, ...}}}
        data_loader = self.data_loader
        exp_result_dict = {}
        fidelity_all = []
        for i, slc in tqdm(self.data_loader.iter_daily(), total=self.data_loader.daily_length):
            feature, label, market_value, stock_index, index = data_loader.get(slc)
            date = datetime.datetime.date(index[0][0])
            graph = self.graph_data[stock_index][:, stock_index]
            dgl_graph = self.explainer.dense2dgl(graph, feature, self.explainer.device)
            if self.explainer_name == 'xpathExplainer':
                exp_result_dict[str(date)] = {}
                if not stock_list:
                    stock_id_list = torch.arange(len(stock_index))
                else:
                    stock_codes = index.get_level_values(1).unique().tolist()
                    stock_id_list = [stock_codes.index(x) for x in stock_list]
                original_pred = self.pred_model(feature, graph).detach().cpu().numpy()
                for stock_id in stock_id_list:
                    if get_fidelity:
                        explanation, fidelity = \
                            self.explainer.explain_dense(self.pred_model, original_pred, dgl_graph, graph,
                                                              stock_id, get_fidelity=get_fidelity, top_k=top_k)
                        fidelity_all.append(fidelity)
                    else:
                        explanation = \
                            self.explainer.explain_dense(self.pred_model, original_pred, dgl_graph, graph,
                                                              stock_id, top_k=top_k)
                    res = {}
                    for k, v in explanation.items():
                        k_stock = index[k][1]
                        res[k_stock] = {}
                        res[k_stock]['score'] = v
                        if relation_list:
                            stock_relations = graph[stock_id, k, :].nonzero().squeeze().tolist()
                            res[k_stock]['relations'] = np.array(relation_list)[stock_relations].tolist()
                            if type(res[k_stock]['relations']) == str:
                                res[k_stock]['relations'] = [res[k_stock]['relations']]
                    exp_result_dict[str(date)][index[stock_id][1]] = res
        if get_fidelity:
            return exp_result_dict, np.mean(fidelity_all)
        return exp_result_dict

    def save_explanation(self):
        file = r'{}/{}-{}.pkl'.format(self.args.expl_results_dir, self.explainer_name, self.year)
        f = open(file, 'wb')
        pickle.dump(self.explained_graph_dict[self.explainer_name], f)
        f.close()
        print('Save Explanation {}'.format(file))

    def save_evaluation(self):
        self.eval_results_df = pd.DataFrame(self.evaluation_results, columns=['fidelity_L1', 'causality_L1'])
        self.eval_results_df.to_csv(r'{}/{}_{}.csv'.format(r'./result',
                                                           self.explainer_name, self.year), index=False)

    def get_eval_results_df(self):
        self.eval_results_df = pd.DataFrame(self.evaluation_results, columns=['fidelity_L1', 'causality_L1'])

    def evaluate(self, original_pred, graph, feature, label):
        top_k_graph, top_k_comp_graph = self.edges_extraction(graph)
        pred_top_k = self.pred_model(feature, top_k_graph)
        pred_top_k_comp = self.pred_model(feature, top_k_comp_graph)

        fidelity_list = Explanation.cal_fidelity(pred_top_k, pred_top_k_comp, original_pred)
        causality_list = Explanation.cal_causality(pred_top_k_comp, original_pred, label)

        return fidelity_list.tolist(), causality_list.tolist()

    def edges_extraction(self, origin_graph):
        sorted_edges = sorted(self.EG.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
        top = range(0, math.ceil(int(len(sorted_edges)) * self.args.top_p))
        top_k_edges = [sorted_edges[x] for x in top]
        top_k_comp_edges = [sorted_edges[x] for x in range(0, len(sorted_edges)) if x not in top]

        top_k_graph = Explanation.edges_2_graph(top_k_comp_edges, origin_graph.clone().detach())
        top_k_comp_graph = Explanation.edges_2_graph(top_k_edges, origin_graph.clone().detach())
        return top_k_graph, top_k_comp_graph

    def cal_random_selection_metrics(self, original_pred, label, feature, random_k_complement, random_k_graph):
        pred_top_k_comp = self.pred_model(feature, random_k_complement)
        pred_top_k = self.pred_model(feature, random_k_graph)
        fidelity_l1 = Explanation.cal_fidelity(pred_top_k_comp, pred_top_k, original_pred)
        causality_L1 = Explanation.cal_causality(pred_top_k_comp, original_pred, label)

        return float(fidelity_l1), float(causality_L1)

    def random_edges_extraction(self, graph):
        new_graph = torch.zeros([graph.shape[0], graph.shape[1]], dtype=torch.int)
        mask = torch.sum(graph, 2)  # mask that could have relation value
        index = torch.t((mask == 1).nonzero())
        new_graph[index[0], index[1]] = 1
        G_edge = nx.from_numpy_array(new_graph.detach().numpy())
        G_edge = list(G_edge.edges)
        edge_num = len(G_edge)
        ran = random.sample(range(0, edge_num), math.ceil(edge_num * self.args.top_k))
        ran_k_edges = [G_edge[x] for x in ran]
        ran_k_comp_edges = [G_edge[x] for x in range(0, edge_num) if x not in ran]

        ran_k_graph = Explanation.edges_2_graph(ran_k_comp_edges, graph.clone().detach())
        ran_k_comp_graph = Explanation.edges_2_graph(ran_k_edges, graph.clone().detach())
        return ran_k_graph, ran_k_comp_graph

    def cal_mean_evaluation(self):
        mean_fidelity_L1 = self.eval_results_df['fidelity_L1'].mean()
        mean_causality_L1 = self.eval_results_df['causality_L1'].mean()
        return mean_fidelity_L1, mean_causality_L1

    def evaluate_explanation(self, feature, explanation_matrix, origin_matrix, p=0.2):
        feature = torch.tensor(feature) if not torch.is_tensor(feature) else feature
        origin_matrix = torch.tensor(origin_matrix) if not torch.is_tensor(origin_matrix) else origin_matrix
        explanation_matrix = torch.tensor(explanation_matrix) if not torch.is_tensor(explanation_matrix) else explanation_matrix
        top_p_matrix = Explanation.select_top_pers_edge(explanation_matrix, p)
        top_p_comp_matrix = origin_matrix - top_p_matrix
        pred_top_k = self.pred_model(feature, top_p_matrix)
        pred_top_k_comp = self.pred_model(feature, top_p_comp_matrix)
        pred_origin = self.pred_model(feature, origin_matrix)


        fidelity = Explanation.cal_fidelity(pred_top_k, pred_top_k_comp, pred_origin)
        return fidelity

    @staticmethod
    # matrix 的维度为3

    def select_top_pers_edge(matrix, p):
        # 将第三个维度加起来，得到一个二维矩阵
        n_matrix = torch.sum(matrix, dim=2)

        # 将输入矩阵中有值的地方全部换成1
        binary_matrix = torch.where(matrix > 0, torch.tensor(1.0), torch.tensor(0.0))

        # 找出所有非零边的索引
        non_zero_indices = torch.nonzero(n_matrix)
        # 获得非零边的权重，并在找出前20%的边
        non_zero_weights = n_matrix[non_zero_indices[:, 0], non_zero_indices[:, 1]]
        k = int(non_zero_weights.numel() * p)
        topk_values, tops_indices = torch.topk(non_zero_weights, k)

        # 创建全零张量
        top_pers_edges = torch.zeros_like(n_matrix)

        # 将对应的位置扩充并填充1
        top_pers_edges[non_zero_indices[tops_indices, 0], non_zero_indices[tops_indices, 1]] = 1

        # 重建原来的三维矩阵，将三维矩阵对应为0的位置全部换成0
        top_pers_edges_expanded = top_pers_edges.unsqueeze(2).expand_as(matrix)
        result_matrix = binary_matrix * top_pers_edges_expanded

        return result_matrix

    @staticmethod
    def select_top_k_related_stock(relative_stocks_dict, k=3):
        new_relative_stocks_dict = {}
        for d, s in relative_stocks_dict.items():
            stock_dict = {}
            for os, rs in s.items():
                if len(rs.items()) <= k:
                    k = len(rs.items())
                sorted_rs = sorted(rs.items(), key=lambda x: x[1], reverse=True)[:k]
                stock_dict[os] = sorted_rs
            new_relative_stocks_dict[d] = stock_dict
        return new_relative_stocks_dict

    @staticmethod
    def nx_to_w_adj(G, origin_graph):
        graph = np.zeros([origin_graph.shape[0], origin_graph.shape[0]])
        for edge in G.edges:
            graph[edge[0], edge[1]] = G.adj[edge[0]][edge[1]]['weight']
        new_G = nx.from_numpy_array(graph)
        return new_G

    @staticmethod
    def cal_fidelity(top_k_pred, top_k_com_pred, original_pred):
        # L1
        loss_func = torch.nn.L1Loss(reduction='none')
        fidelity_s = loss_func(top_k_pred, original_pred).detach().numpy()
        fidelity_c = loss_func(top_k_com_pred, original_pred).detach().numpy()
        fidelity = fidelity_c - fidelity_s
        fidelity[fidelity <= 0] = 0
        mean_fidelity = np.mean(fidelity)
        return mean_fidelity

    @staticmethod
    def cal_causality(top_k_comp_pred, original_pred, label):
        # L1
        loss_func = torch.nn.L1Loss(reduction='none')
        tok_k_comp_l1 = loss_func(top_k_comp_pred, label).detach().numpy()
        original_l1 = loss_func(original_pred, label).detach().numpy()
        causality_array = 1 - original_l1 / tok_k_comp_l1
        return causality_array

    @staticmethod
    def edges_2_graph(comp_edges, origin_graph):
        for edge in comp_edges:
            origin_graph[edge[0], edge[1]] = 0
        return origin_graph


def stock_2_index(stock, stock_index_in_adj, stock_index):
    index = stock_index[stock]
    return stock_index_in_adj.index(index)


def index_2_stock(index, stock_index_in_adj, stock_index):
    index = stock_index_in_adj[index]
    stock = [x for x in stock_index.keys() if stock_index[x] == index]
    return stock[0]


def v_2_key(dict, v):
    values = list(dict.values())
    key = list(dict.keys())
    return key[values.index(v)]

# def parse_args(param_dict):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--device', default='cpu')
#     # model
#     parser.add_argument('--model_name', default=param_dict['model_name'])
#     parser.add_argument('--model_dir', default=param_dict['model_dir'])
#     # explanation
#     parser.add_argument('--expl_results_dir', type=str, default='./ExplanationResults')
#     parser.add_argument('--save_name', type=str, default='pred_loss2021')
#     parser.add_argument('--init_strategy', type=str, default='normal')
#     parser.add_argument('--mask_act', type=str, default='sigmoid')
#     parser.add_argument('--opt', type=str, default='adam')
#
#     parser.add_argument('--explainer', type=str, default='inputGradientExplainer')
#     parser.add_argument('--lr', type=float, default=0.1)
#     parser.add_argument('--size_lamda', type=float, default=0.000001)
#     parser.add_argument('--density_lamda', type=float, default=0.01)
#     parser.add_argument('--num_epochs', type=int, default=25)
#     parser.add_argument('--top_k', type=float, default=0.2)
#
#     # date
#     parser.add_argument('--data_set', type=str, default='csi300')
#     parser.add_argument('--pin_memory', action='store_false', default=True)
#     parser.add_argument('--batch_size', type=int, default=-1)  # -1 indicate daily batch
#     parser.add_argument('--least_samples_num', type=float, default=1137.0)
#     parser.add_argument('--label', default='')  # specify other labels
#     parser.add_argument('--start_date', default='2021-01-03')
#     parser.add_argument('--end_date', default='2021-01-05')
#
#     # data
#     parser.add_argument('--data_path', type=str, default=r'./data/alpha360.pkl')
#     parser.add_argument('--market_value_path', type=str, default=r'./data/csi300_market_value_07to20.pkl')
#     parser.add_argument('--stock_index', type=str, default=r'./data/csi300_stock_index.npy')
#     parser.add_argument('--graph_data_path', default=param_dict['graph_data_path'])
#     parser.add_argument('--d_feat', type=int, default=param_dict['d_feat'])
#     parser.add_argument('--num_layers', type=int, default=param_dict['num_layers'])
#
#     # check
#     parser.add_argument('--stock_list', type=list, default=[])
#     parser.add_argument('--date_list', type=list, default=[])
#     parser.add_argument('--reserve_top_k_relative_stock', type=int, default=3)
#
#     args = parser.parse_args()
#     return args


# if __name__ == '__main__':
#     args = parse_args(config.NRSR_dict)
#     args.start_date = '2021-01-03'
#     args.end_date = '2021-01-20'
#     args.explainer = 'inputGradientExplainer'
#     args.stock_list = ['SH600010']
#     args.date_list = ['2021-01-04', '2021-01-05']
#     explanation = Explanation(args, explainer_name='InputGradientExplainer')
#     explanation.explain()
#     relative_stocks_dict = explanation.check_relative_stock()
#     score_dict = explanation.check_assessment_score()
#     print(relative_stocks_dict)
#     print(score_dict)
