import datetime

import numpy as np
import pandas as pd
import copy
import torch.nn as nn

import torch
import torch.optim as optim

from .gat import GATModel
from .simplehgn import SimpleHeteroHGN
from .rsr import RSRModel


DK_I = "infer"
DK_L = "learn"

class Graphs(nn.Module):
    """Graphs Model
    Parameters
    ----------
    lr : float
        learning rate
    d_feat : int
        input dimensions for each time step
    metric : str
    metric : str
    metric : str
        the evaluation metric used in early stop
    optimizer : str
        optimizer name
    GPU : int
        the GPU ID used for training
    """

    def __init__(
            self,
            d_feat=6,
            hidden_size=64,
            num_layers=2,
            dropout=0.0,
            n_epochs=200,
            lr=0.001,
            metric="",
            early_stop=20,
            loss="mse",
            base_model="GRU",
            graph_model='GAT',
            model_path=None,
            optimizer="adam",
            GPU=0,
            n_jobs=10,
            seed=None,
            rel_encoding=None,
            stock_name_list=None,
            use_residual=False,
            num_graph_layer=2,
            logger=None,
            **kwargs
    ):
        super().__init__()
        self.logger = logger
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.base_model = base_model
        self.graph_model = graph_model
        self.model_path = model_path
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.n_jobs = n_jobs
        self.stock_name_list = stock_name_list
        self.num_graph_layer = num_graph_layer

        self.logger.info(
            "{}s parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\nbase_model : {}"
            "\nmodel_path : {}"
            "\ndevice : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(graph_model,
                                 d_feat,
                                 hidden_size,
                                 num_layers,
                                 dropout,
                                 n_epochs,
                                 lr,
                                 metric,
                                 early_stop,
                                 optimizer.lower(),
                                 loss,
                                 base_model,
                                 model_path,
                                 self.device,
                                 self.use_gpu,
                                 seed,
                                 )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        if self.graph_model == 'GAT':
            self.model = GATModel(
                d_feat=self.d_feat,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,  # RNN layer
                dropout=self.dropout,
                base_model=self.base_model,
                num_graph_layer=self.num_graph_layer,
                use_residual=use_residual
            )
        elif self.graph_model == 'simpleHGN':
            self.model = SimpleHeteroHGN(
                d_feat=self.d_feat, edge_dim=8, num_etypes=rel_encoding.shape[-1],
                num_hidden=self.hidden_size,
                num_layers=self.num_layers,  # RNN layer
                dropout=self.dropout,
                feat_drop=0.5,
                attn_drop=0.5,
                negative_slope=0.05,
                graph_layer_residual=True,
                alpha=0.05,
                base_model=self.base_model,
                num_graph_layer=self.num_graph_layer,
                # heads = [8]*self.num_graph_layer,
                use_residual=True
            )

        elif self.graph_model == 'RSR':
            self.model = RSRModel(d_feat=self.d_feat, hidden_size=self.hidden_size, num_etypes=rel_encoding.shape[-1],
                                  num_layers=self.num_layers, dropout=self.dropout, base_model=self.base_model,
                                  use_residual=False)

        self.logger.info("model:\n{:}".format(self.graph_model))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.model.to(self.device)
        self.model.set_graph(rel_encoding=rel_encoding, device=self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):

        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def cal_my_IC(self, prediction, label):
        pd_label = pd.Series(label.detach().cpu().numpy().squeeze())
        pd_prediction = pd.Series(prediction.detach().cpu().numpy().squeeze())
        return pd_prediction.corr(pd_label)

    def get_daily_inter(self, df, shuffle=False, step_size=1):
        # organize the train data into daily batches
        daily_count = df.groupby(level=0).size().values
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0

        stocks_daily_ = []
        date_list = []
        for date, stocks in pd.Series(index=df.index, dtype=np.float32).groupby("datetime"):
            stocks_daily_.append(list(stocks.loc[date, :].index))
            date_list.append(str(datetime.datetime.date(date)))
        indexes = np.array([i for i in range(len(daily_index))])
        if shuffle:
            np.random.shuffle(indexes)
        daily_index = daily_index[indexes]
        daily_count = daily_count[indexes]
        stocks_daily = [stocks_daily_[i] for i in indexes]

        # sample the dates by step_size
        daily_index = daily_index[::step_size]
        daily_count = daily_count[::step_size]
        stocks_daily = stocks_daily[::step_size]

        return date_list, daily_index, daily_count, stocks_daily

    def train_epoch(self, x_train, y_train):
        self.model.train()
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)

        date_list, daily_index, daily_count, stocks_daily = self.get_daily_inter(x_train, shuffle=True)

        for idx, count, stks in zip(daily_index, daily_count, stocks_daily):
            index = [self.stock_name_list.index(stk) for stk in stks]
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_train_values[batch]).float().to(self.device)
            label = torch.from_numpy(y_train_values[batch]).float().to(self.device)

            pred = self.model(feature.float(), index)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)
        self.model.eval()

        scores = []
        losses = []
        my_ICs = []
        # organize the test data into daily batches
        daily_index, daily_count, stocks_daily = self.get_daily_inter(data_x, shuffle=False)

        for idx, count, stks in zip(daily_index, daily_count, stocks_daily):
            index = [self.stock_name_list.index(stk) for stk in stks]
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)
            label = torch.from_numpy(y_values[batch]).float().to(self.device)

            with torch.no_grad():
                pred = self.model(feature.float(), index)
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())
            IC = self.cal_my_IC(pred, label)
            my_ICs.append(IC)

            score = self.metric_fn(pred, label)
            scores.append(score.item())
        return np.nanmean(losses), np.nanmean(scores), np.nanmean(my_ICs)

    def fit(
            self,
            dataset,
            evals_result=dict(),
            save_path=None,
    ):

        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DK_L,
        )

        df_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DK_I)

        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]
        x_test, y_test = df_test["feature"], df_test["label"]

        # save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(x_train, y_train)
            self.logger.info("evaluating...")
            train_loss, train_score, train_IC = self.test_epoch(x_train, y_train)
            val_loss, val_score, val_IC = self.test_epoch(x_valid, y_valid)
            test_loss, test_score, test_IC = self.test_epoch(x_test, y_test)
            self.logger.info("train %.6f, valid %.6f, test %.4f" % (train_score, val_score, test_score))
            self.logger.info("train IC %.6f, valid IC %.6f, test IC %.4f" % (train_IC, val_IC, test_IC))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def load_checkpoint(self, save_path):
        best_param = torch.load(save_path, map_location=self.device)
        self.model.load_state_dict(best_param)
        self.fitted = True

    def predict(self, dataset, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature")
        df_index = x_test.index
        self.model.eval()
        x_values = x_test.values
        preds = []

        daily_index, daily_count, stocks_daily = self.get_daily_inter(x_test, shuffle=False)

        for idx, count, stks in zip(daily_index, daily_count, stocks_daily):
            index = [self.stock_name_list.index(stk) for stk in stks]
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)

            with torch.no_grad():
                pred = self.model(feature.float(), index).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=df_index)

    def get_fidelity(self, explanation_graph, newstkid, pred):
        new_pred = self.model.predict_on_graph(explanation_graph).detach().cpu().numpy()
        # print(explanation_graph.num_nodes(), new_pred[newstkid], pred)
        fidelity = abs(new_pred[newstkid] - pred)
        return fidelity

    def get_explanation(self, data_df, explainer, stocks, step_size=10, top_k=5, cached_explanations=None, debug=False):
        '''
        stocks is a list of stocks that need to be explained. If None, generate explanation for all batch stocks.
        '''

        x_explain = data_df.iloc[:, :-1]
        all_labels = data_df.iloc[:,-1]
        self.model.eval()
        x_values = x_explain.values
        all_labels = all_labels.values
        date_list, daily_index, daily_count, stocks_daily = self.get_daily_inter(x_explain, shuffle=False, step_size=step_size)
        explanations = {}
        scores_mask = []
        xsize = []

        for i, (date, idx, count, stks) in enumerate(zip(date_list, daily_index, daily_count, stocks_daily)):
            index = [self.stock_name_list.index(stk) for stk in stks]
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)
            labels = torch.from_numpy(all_labels[batch]).float().to(self.device)
            batch_explanations = {} if cached_explanations is None else cached_explanations[i]
            batch_fidelity_mask = 0
            index_to_explain = []
            batch_graph_size = 0
            with torch.no_grad():
                if stocks:
                    for stk in stocks:
                        try:
                            tmp = self.stock_name_list.index(stk)
                        except:
                            self.logger.info(f"{stk} is not in this batch")
                        index_to_explain.append([tmp, stk])
                else:
                    index_to_explain = zip(index, stks)

                graph_feature = self.model.forward_rnn(feature.float())
                graph_emb, subgraph = self.model.forward_graph(graph_feature, index, return_subgraph=True)
                # subgraph is the graph of the batch stocks (stocks in this timestamp)
                pred = self.model.forward_predictor(graph_feature, graph_emb).detach().cpu().numpy()
                subgraph.ndata['nfeat'] = graph_feature
                original_edges = subgraph.number_of_edges()

                exp_edges = 0
                f_score_list = []
                a_score_list = []
                # index:N -> [0,1...N-1]
                # There are g, subgraph, explanation_graph, please note the index change
                for i, (stkid, stk) in enumerate(index_to_explain): # stkid is the in g, not the one if subgraph
                    id_sub = index.index(stkid)
                    if cached_explanations is None:
                        explanation = explainer.explain(self.model, subgraph, id_sub)
                        batch_explanations[stk] = explanation  # note the node index in explanation is in subgraph!
                    else:
                        try:
                            explanation = batch_explanations[stk]
                        except:
                            continue

                    # 计算fidelity
                    explanation_graph_mask, id_xgraph_mask = \
                        explainer.explanation_to_graph(explanation, subgraph, id_sub, top_k=top_k,
                                                       maskout=False)
                    fidelity_n = self.get_fidelity(explanation_graph_mask, id_xgraph_mask, pred[id_sub])

                    explanation_graph_mask_c, id_xgraph_mask = \
                        explainer.explanation_to_graph(explanation, subgraph, id_sub, top_k=top_k,
                                                       maskout=True)
                    fidelity_c = self.get_fidelity(explanation_graph_mask_c, id_xgraph_mask, pred[id_sub])
                    if fidelity_c == 0:
                        fidelity_c = 0.00001
                    f_score = 1 - fidelity_n/fidelity_c
                    if f_score < 0:
                        f_score = 0
                    f_score_list.append(f_score)

                    # 计算accuracy
                    original_pred = pred[id_sub]
                    new_pred = self.model.predict_on_graph(explanation_graph_mask_c).detach().cpu().numpy()[id_xgraph_mask]
                    label = labels[id_sub]

                    a_score = float(1 - abs(original_pred-label)/abs(new_pred-label))
                    if a_score < 0:
                        a_score = 0
                    a_score_list.append(a_score)

                    exp_edges += explanation_graph_mask.number_of_edges()


                    # explanation_graph_mask, id_xgraph_mask = explainer.explanation_to_graph(explanation, subgraph, id_sub, top_k=top_k)
                    # fidelity_mask = self.get_fidelity(explanation_graph_mask, id_xgraph_mask, pred[id_sub])

                    # batch_fidelity_mask += fidelity_mask
                    batch_graph_size += explanation_graph_mask.num_nodes()

                s_score = exp_edges/original_edges
                scores_dict = {}
                idx = 0
                for stock, exps in batch_explanations.items():
                    scores_dict[stock] = {}
                    scores_dict[stock]['f_score'] = f_score_list[idx] * 5
                    scores_dict[stock]['a_score'] = a_score_list[idx] * 5
                    scores_dict[stock]['s_score'] = s_score * 5
                    scores_dict[stock]['score'] = scores_dict[stock]['f_score'] + scores_dict[stock]['a_score'] + scores_dict[stock]['s_score']
                    idx += 1

                    ne_re = {}
                    for k , v in exps.items():
                        ne_re[stks[k[1]]] = v
                    batch_explanations[stock] = ne_re

                explanations[date] = batch_explanations
                num_stock = len(batch_explanations.keys())
                if num_stock>0:
                    fidelity_mask_score = batch_fidelity_mask/num_stock
                    graph_size = batch_graph_size/num_stock
                    scores_mask.append(fidelity_mask_score)
                    xsize.append(graph_size)
                if debug:
                    self.logger.info(f" Explain for {num_stock} stocks in batch, "
                                     f"fidelity mask score {fidelity_mask_score}, xgraph size {graph_size}..")
        if debug:
            self.logger.info(f" Overall fidelity mask score {sum(scores_mask)/len(scores_mask)}.")
            self.logger.info(f" Overall xgraph size {sum(xsize)/len(xsize)}.")



        return explanations, scores_mask, scores_dict

    def get_one_explanation(self, data_df, stocks,  explainer, top_k):
        '''
        Get explanation for one stocks from start time to end time.
        '''
        x_explain = data_df.iloc[:, :-1]
        self.model.eval()
        x_values = x_explain.values
        date_list, daily_index, daily_count, stocks_daily = self.get_daily_inter(x_explain, shuffle=False, step_size=1)
        explanations = {}

        for i, (date, idx, count, stks) in enumerate(zip(date_list, daily_index, daily_count, stocks_daily)):
            index = [self.stock_name_list.index(stk) for stk in stks]
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)
            batch_explanations = {}
            index_to_explain = []
            with torch.no_grad():
                if stocks:
                    for stk in stocks:
                        try:
                            tmp = self.stock_name_list.index(stk)
                        except:
                            self.logger.info(f"{stk} is not in this batch")
                        index_to_explain.append([tmp, stk])
                else:
                    index_to_explain = zip(index, stks)

                graph_feature = self.model.forward_rnn(feature.float())
                graph_emb, subgraph = self.model.forward_graph(graph_feature, index, return_subgraph=True)
                subgraph.ndata['nfeat'] = graph_feature

                for i, (stkid, stk) in enumerate(index_to_explain):
                    try:
                        id_sub = index.index(stkid)
                        explanation = explainer.explain(self.model, subgraph, id_sub)
                        if explainer.__class__.__name__ == 'xPath':
                            explanation_sorted = sorted(explanation.items(), key=lambda d: d[1], reverse=True)
                            explanation_orig = explanation_sorted[:top_k] \
                                if top_k <= len(explanation_sorted) else explanation_sorted
                            explanation_processed = [((self.stock_name_list[index[x[0][0]]],
                                                       self.stock_name_list[index[x[0][1]]]),
                                                      x[1]) for x in explanation_orig]
                        elif explainer.__class__.__name__ == 'AttentionX':
                            explanation_sorted = sorted(explanation, key=lambda d: d[1], reverse=True)
                            explanation_orig = explanation_sorted[:top_k] \
                                if top_k <= len(explanation_sorted) else explanation_sorted
                            explanation_processed = [[self.stock_name_list[index[x[0]]], x[1]]
                                                     for x in explanation_orig]
                        elif explainer.__class__.__name__ == 'SubgraphXExplainer':
                            # find the most suitable subgraph with the highest reward
                            explanation_sorted = sorted(explanation, key=lambda d: d[1], reverse=True)
                            explanation_processed = None
                            for x in explanation_sorted:
                                if len(x[0]) <= top_k:
                                    explanation_processed = [x[0], x[1]]
                                    break
                        else:
                            raise NotImplementedError

                        batch_explanations[stk] = explanation_processed
                    except ValueError:
                        print(f'stock {stocks[i]} has no neighbors in the graph.')
                        batch_explanations[stk] = None
                explanations[date] = batch_explanations

        return explanations

    # def get_one_explanation(self, dataset, stocks, start_time, end_time, explainer, top_k):
    #     '''
    #     Get explanation for one stocks from start time to end time.
    #     '''
    #     x_explain = dataset.prepare_seg((start_time, end_time), col_set='feature', level='datetime', data_key=DK_I)
    #     self.model.eval()
    #     x_values = x_explain.values
    #     daily_index, daily_count, stocks_daily = self.get_daily_inter(x_explain, shuffle=False, step_size=1)
    #     explanations = []
    #
    #     for i, (idx, count, stks) in enumerate(zip(daily_index, daily_count, stocks_daily)):
    #         print(i)
    #         index = [self.stock_name_list.index(stk) for stk in stks]
    #         batch = slice(idx, idx + count)
    #         feature = torch.from_numpy(x_values[batch]).float().to(self.device)
    #         batch_explanations = {}
    #         index_to_explain = []
    #         with torch.no_grad():
    #             if stocks:
    #                 for stk in stocks:
    #                     try:
    #                         tmp = self.stock_name_list.index(stk)
    #                     except:
    #                         self.logger.info(f"{stk} is not in this batch")
    #                     index_to_explain.append([tmp, stk])
    #             else:
    #                 index_to_explain = zip(index, stks)
    #
    #             graph_feature = self.model.forward_rnn(feature.float())
    #             graph_emb, subgraph = self.model.forward_graph(graph_feature, index, return_subgraph=True)
    #             subgraph.ndata['nfeat'] = graph_feature
    #
    #             for i, (stkid, stk) in enumerate(index_to_explain):
    #                 try:
    #                     id_sub = index.index(stkid)
    #                     explanation = explainer.explain(self.model, subgraph, id_sub)
    #                     if explainer.__class__.__name__ == 'xPath':
    #                         explanation_sorted = sorted(explanation.items(), key=lambda d: d[1], reverse=True)
    #                         explanation_orig = explanation_sorted[:top_k] \
    #                             if top_k <= len(explanation_sorted) else explanation_sorted
    #                         explanation_processed = [((self.stock_name_list[index[x[0][0]]],
    #                                                    self.stock_name_list[index[x[0][1]]]),
    #                                                   x[1]) for x in explanation_orig]
    #                     elif explainer.__class__.__name__ == 'AttentionX':
    #                         explanation_sorted = sorted(explanation, key=lambda d: d[1], reverse=True)
    #                         explanation_orig = explanation_sorted[:top_k] \
    #                             if top_k <= len(explanation_sorted) else explanation_sorted
    #                         explanation_processed = [[self.stock_name_list[index[x[0]]], x[1]]
    #                                                  for x in explanation_orig]
    #                     elif explainer.__class__.__name__ == 'SubgraphXExplainer':
    #                         # find the most suitable subgraph with the highest reward
    #                         explanation_sorted = sorted(explanation, key=lambda d: d[1], reverse=True)
    #                         explanation_processed = None
    #                         for x in explanation_sorted:
    #                             if len(x[0]) <= top_k:
    #                                 explanation_processed = [x[0], x[1]]
    #                                 break
    #                     else:
    #                         raise NotImplementedError
    #
    #                     batch_explanations[stk] = explanation_processed
    #                 except ValueError:
    #                     print(f'stock {stocks[i]} has no neighbors in the graph.')
    #                     batch_explanations[stk] = None
    #             explanations.append(batch_explanations)
    #
    #     return explanations
