"""
incremental learning
support: LSTM, ALSTM, GRU, GATS, HIST, SFM, RSR
"""
import datetime
import sys
sys.path.insert(0, sys.path[0]+"/../")
from utils.dataloader import create_incre_loaders
from utils.dataloader import create_incre_pre_loaders
import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
import copy
import argparse
import os
import torch.optim as optim
import collections
from tqdm import tqdm
from models.model import MLP, HIST, GRU, LSTM, GAT, ALSTM, SFM, RSR, relation_GATs, relation_GATs_3heads
from utils.utils import metric_fn, mse, loss_ic, pair_wise_loss, NDCG_loss, ApproxNDCG_loss
from qlib.contrib.model.pytorch_transformer import Transformer
from utils.dataloader import create_loaders
from models.DLinear import DLinear_model
from models.Autoformer import Model as autoformer
from models.Crossformer import Model as crossformer
from models.ETSformer import Model as ETSformer
from models.FEDformer import Model as FEDformer
from models.FiLM import Model as FiLM
from models.Informer import Model as Informer
from models.PatchTST import Model as PatchTST
import json

time_series_library = [
    'DLinear',
    'Autoformer',
    'Crossformer',
    'ETSformer',
    'FEDformer',
    'FiLM',
    'Informer',
    'PatchTST'
]

relation_model_dict = [
    'RSR',
    'relation_GATs',
    'relation_GATs_3heads'
]

global_step = -1
def loss_fn(pred, label):
    mask = ~torch.isnan(label)
    # output is a single dim tensor
    return mse(pred[mask], label[mask])

def get_model(model_name):
    if model_name.upper() == 'MLP':
        return MLP

    if model_name.upper() == 'LSTM':
        return LSTM

    if model_name.upper() == 'GRU':
        return GRU

    if model_name.upper() == 'TRANSFORMER':
        return Transformer

    if model_name.upper() == 'GATS':
        return GAT

    if model_name.upper() == 'SFM':
        return SFM

    if model_name.upper() == 'ALSTM':
        return ALSTM

    if model_name.upper() == 'HIST':
        return HIST

    if model_name.upper() == 'RSR':
        return RSR

    raise ValueError('unknown model name `%s`' % model_name)

def plot_norm(num_bins, losses):
    # mpl.use('TkAgg')
    losses = np.array(losses)
    mu = np.mean(losses)  # 计算均值
    sigma = np.std(losses)
    return mu, sigma


def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError('the %d-th model has different params'%i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params


def test_epoch(model, test_loader, param_dict, stock2concept_matrix=None, stock2stock_matrix=None, prefix='Test'):
    model.eval()
    losses = []
    for i, slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):
        feature, label, market_value, stock_index, index, mask = test_loader.get(slc)
        with torch.no_grad():
            if param_dict['model_name'] == 'HIST':
                pred = model(feature, stock2concept_matrix[stock_index], market_value)
            elif param_dict['model_name'] in relation_model_dict:
                pred = model(feature, stock2stock_matrix[stock_index][:, stock_index])
            elif param_dict['model_name'] in time_series_library:
                # new added
                pred = model(feature, mask)
            else:
                pred = model(feature)
            if param_dict['loss_type'] == 'ic':
                loss = loss_ic(pred, label)
            elif param_dict['loss_type'] == 'pair_wise':
                loss = pair_wise_loss(pred, label)
            elif param_dict['loss_type'] == 'ndcg':
                loss = NDCG_loss(pred, label)
            elif param_dict['loss_type'] == 'appndcg':
                loss = ApproxNDCG_loss(pred, label)
            else:
                loss = loss_fn(pred, label)
        losses.append(loss.item())
    return losses


def incremental_epoch(model, prev_model, optimizer, incre_loader,  args, mu, p, sigma, param_dict, alpha = 0.5,
                stock2concept_matrix=None, stock2stock_matrix = None):

    global global_step
    filter = [0, 0]  # [process, non-process]
    prev_model.train()
    model.train()

    for i, slc in tqdm(incre_loader.iter_batch(), total=incre_loader.batch_length):
        global_step += 1
        feature, label, market_value, stock_index, _, mask = incre_loader.get(slc)
        if param_dict['model_name'] == 'HIST':
            # if HIST is used, take the stock2concept_matrix and market_value
            pred = model(feature, stock2concept_matrix[stock_index], market_value)
            pred_old = prev_model(feature, stock2concept_matrix[stock_index], market_value)
        elif param_dict['model_name'] in relation_model_dict:
            pred = model(feature, stock2stock_matrix[stock_index][:, stock_index])
            pred_old = prev_model(feature, stock2stock_matrix[stock_index][:, stock_index])
        elif param_dict['model_name'] in time_series_library:
            # new added
            pred = model(feature, mask)
            pred_old = prev_model(feature, mask)
        else:
            # other model only use feature as input
            pred = model(feature)
            pred_old = prev_model(feature)

        if param_dict['loss_type'] == 'ic':
            loss = loss_ic(pred, label)
        elif param_dict['loss_type'] == 'pair_wise':
            loss = pair_wise_loss(pred, label)
        elif param_dict['loss_type'] == 'ndcg':
            loss = NDCG_loss(pred, label)
        elif param_dict['loss_type'] == 'appndcg':
            loss = ApproxNDCG_loss(pred, label)
        else:
            loss = loss_fn(pred, label)

        if mu + alpha * sigma >= loss >= mu - alpha * sigma:
            filter[1] = filter[1] + 1
        else:
            filter[0] = filter[0] + 1
            dist_loss = loss_fn(pred, pred_old)
            loss_ = loss + p * dist_loss
            optimizer.zero_grad()
            loss_.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
            optimizer.step()
    # print("filter=[{},{}]".format(filter[0],filter[1]))


def incremental_learning(args, model_path, device):
    param_dict = json.load(open(model_path+'/info.json'))['config']
    param_dict['model_dir'] = model_path
    incre_loader = create_incre_loaders(args, param_dict, device=device)
    incre_pre_loader = create_incre_pre_loaders(args, param_dict, device=device)

    stock2concept_matrix = param_dict['stock2concept_matrix']
    stock2stock_matrix = param_dict['stock2stock_matrix']
    print('load model ', param_dict['model_name'])
    if param_dict['model_name'] == 'SFM':
        model = get_model(param_dict['model_name'])(d_feat=param_dict['d_feat'], output_dim=32, freq_dim=25,
                                                    hidden_size=param_dict['hidden_size'],
                                                    dropout_W=0.5, dropout_U=0.5, device=device)
    elif param_dict['model_name'] == 'ALSTM':
        model = get_model(param_dict['model_name'])(param_dict['d_feat'], param_dict['hidden_size'],
                                                    param_dict['num_layers'], param_dict['dropout'], 'LSTM')
    elif param_dict['model_name'] == 'Transformer':
        model = get_model(param_dict['model_name'])(param_dict['d_feat'], param_dict['hidden_size'],
                                                    param_dict['num_layers'], dropout=0.5)
    elif param_dict['model_name'] == 'HIST':
        # HIST need stock2concept matrix, send it to device
        stock2concept_matrix = torch.Tensor(np.load(stock2concept_matrix)).to(device)
        model = get_model(param_dict['model_name'])(d_feat=param_dict['d_feat'], num_layers=param_dict['num_layers']
                                                    , K=param_dict['K'])
    elif param_dict['model_name'] in relation_model_dict:
        stock2stock_matrix = torch.Tensor(np.load(stock2stock_matrix)).to(device)
        num_relation = stock2stock_matrix.shape[2]  # the number of relations
        model = get_model(param_dict['model_name'])(num_relation=num_relation, d_feat=param_dict['d_feat'],
                                                    num_layers=param_dict['num_layers'])
    else:
        model = get_model(param_dict['model_name'])(d_feat=param_dict['d_feat'], num_layers=param_dict['num_layers'])

    model.to(device)
    model.load_state_dict(torch.load(param_dict['model_dir'] + '/model.bin', map_location=device))

    # compute the mean and sigma of the losses distribution
    losses = test_epoch(model,incre_pre_loader,param_dict,stock2concept_matrix,stock2stock_matrix, prefix='Train')
    mu, sigma = plot_norm(num_bins=40, losses=losses)

    prev_model = copy.deepcopy(model)
    optimizer = optim.Adam(model.parameters(), lr=param_dict['lr'])
    best_score = -np.inf
    best_epoch = 0
    stop_round = 0
    # save best parameters
    best_param = copy.deepcopy(model.state_dict())
    params_list = collections.deque(maxlen=param_dict['smooth_steps'])

    for epoch in range(param_dict['n_epochs']):
        print('Epoch:', epoch)
        print('training...')
        incremental_epoch(model, prev_model, optimizer, incre_loader, args, stock2concept_matrix=stock2concept_matrix,
                          stock2stock_matrix=stock2stock_matrix, p=args.incre_p, mu=mu, sigma=sigma,
                          alpha=args.incre_alpha, param_dict=param_dict)

        params_ckpt = copy.deepcopy(model.state_dict())
        params_list.append(params_ckpt)
        avg_params = average_params(params_list)
        # when evaluating, use the avg_params in the current time to evaluate
        model.load_state_dict(avg_params)

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    # save to the model_save_path(incremental path)
    torch.save(avg_params, args.model_save_path + '/model.bin')


def main(args, device):
    model_path = args.model_path
    incremental_learning(args, model_path, device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--incre_start_date', default='2020-01-01')
    parser.add_argument('--incre_end_date', default='2022-05-31')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model_path', default='./output/for_platform/LSTM', help='learned model')
    parser.add_argument('--model_save_path', default='./output/for_platform/INCRE/LSTM_incre', help='updated model')
    parser.add_argument('--incre_p', default=1000000, help='weight of distillation losses')
    parser.add_argument('--incre_alpha', default=0.1, help='trigger hyperparameters for incremental updates')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    for model_name in ['ALSTM', 'GATs', 'GRU', 'LSTM', 'MLP', 'SFM']:
        args.model_path = './output/for_platform/' + model_name
        args.model_save_path = './output/for_platform/INCRE/' + model_name + '_incre'
        main(args, device)