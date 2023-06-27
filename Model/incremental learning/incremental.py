import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
import json
import argparse
import datetime
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
import qlib
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm
from scipy import stats
import random
# regiodatetimeG_CN, REG_US]
from qlib.config import REG_US, REG_CN
from model import MLP, HIST, GRU, LSTM, GAT, ALSTM, SFM, RSR
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")  # 忽略UserWarning兼容性警告

provider_uri = "~/.qlib/qlib_data/cn_data_build"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
# seed = np.random.randint(1000000)
seed = 441051


#返回当前设备索引
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'




def parse_args():

    parser = argparse.ArgumentParser()

    # model
    model_name = 'MLP'
    parser.add_argument('--model_name', default = model_name)
    parser.add_argument('--d_feat', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--baseline_exist', type=bool, default=False)   # True: load the existing baseline
    parser.add_argument('--basic_exist', type=bool, default=False)
    parser.add_argument('--incremental_exist', type=bool, default=False)

    # training
    parser.add_argument('--n_epochs', type=int, default=200) #default 200
    parser.add_argument('--incre_n_epochs', type=int, default=200) #default 200
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--early_stop', type=int, default=200)
    parser.add_argument('--smooth_steps', type=int, default=40)
    parser.add_argument('--metric', default='IC')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--repeat', type=int, default=10)

    # data
    parser.add_argument('--data_set', type=str, default='csi300')
    parser.add_argument('--pin_memory', action='store_false', default= True)
    parser.add_argument('--batch_size', type=int, default=-1) # DEFAULT = -1 indicate daily batch
    parser.add_argument('--incremental_number_days', type=int, default=-1) # DEFAULT = -1 indicate daily batch
    parser.add_argument('--least_samples_num', type=float, default=1137.0)
    parser.add_argument('--label', default='') # specify other labels
    parser.add_argument('--train_start_date', default='2009-01-01')
    parser.add_argument('--train_end_date', default='2012-12-31')
    parser.add_argument('--incremental_start_date', default='2013-01-01')
    parser.add_argument('--incremental_end_date', default='2013-12-31')
    parser.add_argument('--baseline_start_date', default='2009-01-01')
    parser.add_argument('--baseline_end_date', default='2013-12-31')
    parser.add_argument('--val_start_date', default='2014-01-01')
    parser.add_argument('--val_end_date', default='2014-12-31')
    parser.add_argument('--test_start_date', default='2015-01-01')
    parser.add_argument('--test_end_date', default='2015-12-31')



    # other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--annot', default='')
    parser.add_argument('--config', action=ParseConfigFile, default='')
    # parser.add_argument('--name', type=str, default='csi300_HIST')
    parser.add_argument('--name', type=str, default='csi300_' + model_name)

    # input for csi 300
    parser.add_argument('--market_value_path', default='D:/Fintech_ust/dynamic_updates/model_pool/model_pool/data/csi300_market_value_07to20.pkl')
    parser.add_argument('--stock2concept_matrix', default='D:/Fintech_ust/dynamic_updates/model_pool/model_pool/data/csi300_stock2concept.npy')
    parser.add_argument('--stock2stock_matrix', default='D:/Fintech_ust/dynamic_updates/model_pool/model_pool/data/csi300_multi_stock2stock_all.npy')
    parser.add_argument('--stock_index', default='D:/Fintech_ust/dynamic_updates/model_pool/model_pool/data/csi300_stock_index.npy')

    # parser.add_argument('--outdir', default='./output/csi300_')
    parser.add_argument('--outdir', default='./output/csi300_' + model_name)
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()

    return args


class ParseConfigFile(argparse.Action):
    def __call__(self, parser, namespace, filename, option_string=None):
        if not os.path.exists(filename):
            raise ValueError('cannot find config at `%s`'%filename)

        with open(filename) as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(namespace, key, value)

global_log_file = None
def pprint(*args):
    # print with UTC+8 time
    time = '['+str(datetime.datetime.utcnow()+
                   datetime.timedelta(hours=8))[:19]+'] -'
    print(time, *args, flush=True)

    if global_log_file is None:
        return
    with open(global_log_file, 'a') as f:
        print(time, *args, flush=True, file=f)

def mse(pred, label):
    loss = (pred - label)**2
    return torch.mean(loss)

def get_model(model_name):

    if model_name.upper() == 'MLP':
        return MLP

    if model_name.upper() == 'LSTM':
        return LSTM
        # return LSTMModel

    if model_name.upper() == 'GRU':
        return GRU
        # return GRUModel

    if model_name.upper() == 'GAT':
        return GAT

    if model_name.upper() == 'SFM':
        return SFM

    if model_name.upper() == 'ALSTM':
        return ALSTM

    if model_name.upper() == 'HIST':
        return HIST

    if model_name.upper() == 'RSR':
        return RSR

    raise ValueError('unknown model name `%s`'%model_name)


class DataLoader:

    def __init__(self, df_feature, df_label, df_market_value, df_stock_index, batch_size=800, incremental_number_days = -1, pin_memory=True, start_index = 0, device=None):

        assert len(df_feature) == len(df_label)

        self.df_feature = df_feature.values
        self.df_label = df_label.values
        self.df_market_value = df_market_value
        self.df_stock_index = df_stock_index
        self.device = device

        if pin_memory:
            self.df_feature = torch.tensor(self.df_feature, dtype=torch.float, device=device)
            self.df_label = torch.tensor(self.df_label, dtype=torch.float, device=device)
            self.df_market_value = torch.tensor(self.df_market_value, dtype=torch.float, device=device)
            self.df_stock_index = torch.tensor(self.df_stock_index, dtype=torch.long, device=device)

        self.index = df_label.index

        self.batch_size = batch_size
        self.incremental_number_days = incremental_number_days
        self.pin_memory = pin_memory
        self.start_index = start_index

        self.daily_count = df_label.groupby(level=0).size().values  # level = 0 -> calculate #stocks for each day
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0


    @property
    def batch_length(self):

        if self.batch_size <= 0:
            # this is the default situation
            return self.daily_length

        return len(self.df_label) // self.batch_size

    @property
    def daily_length(self):
        """
        :return:  number of days in the dataloader
        """
        return len(self.daily_count)


    def get(self, slc):
        outs = self.df_feature[slc], self.df_label[slc][:,0], self.df_market_value[slc], self.df_stock_index[slc]
        # outs = self.df_feature[slc], self.df_label[slc]

        if not self.pin_memory:
            outs = tuple(torch.tensor(x, device=self.device) for x in outs)

        return outs + (self.index[slc],)

    def iter_daily_shuffle(self):
        indices = np.arange(len(self.daily_count))
        np.random.shuffle(indices)
        # print("iter_daily_shuffle indices:{}".format(indices))
        # print("daily_count:{}".format(self.daily_count))
        for i in indices:
            # daily_index[i] returns the start day index of day i, daily_count[i] return the #stocks of day i
            yield i, slice(self.daily_index[i], self.daily_index[i] + self.daily_count[i])

    def iter_daily(self):
        """
        : yield an index and a slice, that from the day
        """
        indices = np.arange(len(self.daily_count))
        for i in indices:
            yield i, slice(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        # for idx, count in zip(self.daily_index, self.daily_count):
        #     yield slice(idx, idx + count) # NOTE: slice index will not cause copy

    def iter_incremental(self):
        if self.incremental_number_days <= 0:    # by default
            # yield from self.iter_daily()
            yield from self.iter_daily()
            return
        # len(df_label) is #stocks each day * #days
        indices = np.arange(len(self.daily_count))
        for i in indices[::self.incremental_number_days]:
            num_index = 0
            for j in range(self.incremental_number_days):
                if i + j >= len(self.daily_count):
                    break
                num_index += self.daily_count[i + j]
            yield i, slice(self.daily_index[i], self.daily_index[i] + num_index)



    def iter_batch(self):
        if self.batch_size <= 0:    # by default
            yield from self.iter_daily()
            return
        indices = np.arange(len(self.df_label))
        # np.random.shuffle(indices)

        for i in range(len(indices))[::self.batch_size]:    # batch_size是间隔
            if len(indices) - i < self.batch_size:
                break
            yield i, indices[i:i+self.batch_size] # NOTE: advanced indexing will cause copy


def weight_init(m):
    set_seed(seed)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if args.model_name != 'ALSTM':
            m.bias.data.fill_(0.01)
        # nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GRU):
        for layer in m._all_weights:
            for p in layer:
                if 'weight' in p:
                    # print(p, a.__getattr__(p))
                    nn.init.normal_(m.__getattr__(p), 0.0, 0.02)
                    # print(p, a.__getattr__(p))
                elif 'bias' in p:
                    nn.init.uniform_(getattr(m,p))
    elif isinstance(m, nn.LSTM):
        for layer in m._all_weights:
            for p in layer:
                if 'weight' in p:
                    # print(p, a.__getattr__(p))
                    nn.init.normal_(m.__getattr__(p), 0.0, 0.02)
                    # print(p, a.__getattr__(p))
                elif 'bias' in p:
                    nn.init.uniform_(getattr(m,p))
    elif isinstance(m, nn.Parameter):
        for p in m.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

def loss_fn(pred, label):
    mask = ~torch.isnan(label)
    # output is a single dim tensor
    return mse(pred[mask], label[mask])


def metric_fn(preds):
    preds = preds[~np.isnan(preds['label'])]
    precision = {}
    recall = {}
    temp = preds.groupby(level='datetime').apply(lambda x: x.sort_values(by='score', ascending=False))
    if len(temp.index[0]) > 2:
        temp = temp.reset_index(level=0).drop('datetime', axis=1)

    for k in [1, 3, 5, 10, 20, 30, 50, 100]:
        precision[k] = temp.groupby(level='datetime').apply(lambda x: (x.label[:k] > 0).sum() / k).mean()
        recall[k] = temp.groupby(level='datetime').apply(lambda x: (x.label[:k] > 0).sum() / (x.label > 0).sum()).mean()

    ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score)).mean()
    rank_ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score, method='spearman')).mean()

    return precision, recall, ic, rank_ic

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


def test_epoch(model, test_loader, args, stock2concept_matrix=None, stock2stock_matrix=None, prefix='Test'):
    """
    :return: loss -> mse
             scores -> ic
             rank_ic
             precision, recall -> precision, recall @1, 3, 5, 10, 20, 30, 50, 100
    """
    # 对于Batch Normalization来说，保证Batch Normalization层能够用到每一批数据的均值和方差，并进行计算更新。
    # model.eval()是保证Batch Normalization层直接利用之前训练阶段得到的均值和方差，停止计算和更新mean和val，
    # 所以在测试过程中要保证Batch Normalization层的均值和方差不变
    model.eval()
    losses = []
    preds = []
    for i, slc in test_loader.iter_daily():
        feature, label, market_value, stock_index, index = test_loader.get(slc)
        with torch.no_grad():
            with torch.no_grad():
                if args.model_name == 'HIST':
                    pred = model(feature, stock2concept_matrix[stock_index], market_value)
                elif args.model_name == 'RSR':
                    pred = model(feature, stock2stock_matrix[stock_index][:, stock_index])
                else:
                    pred = model(feature)
            loss = loss_fn(pred, label)
            preds.append(pd.DataFrame({'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))
        losses.append(loss.item())
    # evaluate
    preds = pd.concat(preds, axis=0)
    # use metric_fn to compute precision, recall, ic and rank ic
    precision, recall, ic, rank_ic = metric_fn(preds)
    scores = ic
    # scores = (precision[3] + precision[5] + precision[10] + precision[30])/4.0
    # scores = -1.0 * mse
    return np.mean(losses), scores, precision, recall, ic, rank_ic, losses

def valid_epoch(model, valid_loader, args, stock2concept_matrix=None, stock2stock_matrix=None, prefix='Valid'):
    model.eval()
    losses = []
    preds = []
    for i, slc in valid_loader.iter_batch():
        feature, label, market_value, stock_index, index = valid_loader.get(slc)
        with torch.no_grad():
            with torch.no_grad():
                if args.model_name == 'HIST':
                    pred = model(feature, stock2concept_matrix[stock_index], market_value)
                elif args.model_name == 'RSR':
                    pred = model(feature, stock2stock_matrix[stock_index][:, stock_index])
                else:
                    pred = model(feature)
            loss = loss_fn(pred, label)
            preds.append(pd.DataFrame({'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))
        losses.append(loss.item())
    # evaluate
    preds = pd.concat(preds, axis=0)
    # use metric_fn to compute precision, recall, ic and rank ic
    precision, recall, ic, rank_ic = metric_fn(preds)
    scores = ic
    return np.mean(losses), scores, precision, recall, ic, rank_ic

# p is hyperparameter
def incremental_epoch(model, prev_model, incremental_loader, args, mu, p, sigma, alpha = 0.5, stock2concept_matrix=None, stock2stock_matrix=None, prefix='Increment'):

    best_score = -np.inf
    best_param = copy.deepcopy(model.state_dict())
    params_list = collections.deque(maxlen=args.smooth_steps)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    t1 = time.time()
    for epoch in tqdm(range(args.incre_n_epochs), total = args.incre_n_epochs):
        model.train()
        prev_model.train()
        filter = [0, 0]  # [process, non-process]

        for i, slc in incremental_loader.iter_incremental():

            feature, label, market_value, stock_index, _ = incremental_loader.get(slc)

            if args.model_name == 'HIST':
                pred_old = prev_model(feature, stock2concept_matrix[stock_index], market_value)
                pred_new = model(feature, stock2concept_matrix[stock_index], market_value)
            elif args.model_name == 'RSR':
                pred_old = prev_model(feature, stock2stock_matrix[stock_index][:, stock_index])
                pred_new = model(feature, stock2stock_matrix[stock_index][:, stock_index])
            else:
                pred_old = prev_model(feature)
                pred_new = model(feature)
                # print("pred_old:{}".format(pred_old))
                # print("pred_new:{}".format(pred_new))

            loss_new = loss_fn(pred_new,label)
            # filter
            if loss_new <= mu + alpha * sigma and loss_new >= mu - alpha * sigma:
                filter[1] = filter[1] + 1

            else:
                filter[0] = filter[0] + 1
                dist_loss = loss_fn(pred_new, pred_old)
                loss = loss_new + p * dist_loss
                # loss = loss_new
                # print("distillation loss:{}, newtask loss:{}".format(dist_loss,loss_new))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
                optimizer.step()

        # print("未被过滤:{}, 被过滤:{}".format(filter[0], filter[1]))
        params_ckpt = copy.deepcopy(model.state_dict())
        params_list.append(params_ckpt)
        avg_params = average_params(params_list)
        # when evaluating, use the avg_params in the current time to evaluate
        model.load_state_dict(avg_params)
        # pprint("start validation")
        val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic = valid_epoch(model,
                                                                                          valid_loader,
                                                                                          args, stock2concept_matrix,
                                                                                          stock2stock_matrix,
                                                                                          )
        if val_score > best_score:
            best_param = copy.deepcopy(avg_params)
    t2 = time.time()
    increment_time = t2- t1
    print("incremental time:{}".format(increment_time))
    model.load_state_dict(best_param)
    return increment_time



def train_epoch(model, optimizer, train_loader,  args, stock2concept_matrix = None, stock2stock_matrix = None):
    """
    train epoch function
    :param epoch: number of epoch of training
    :param model: model that will be used
    :param optimizer:
    :param train_loader:
    :param writer:
    :param args:
    :param stock2concept_matrix:
    :return:
    """

    best_score = -np.inf
    stop_round = 0
    best_param = copy.deepcopy(model.state_dict())
    params_list = collections.deque(maxlen=args.smooth_steps)

    t1 = time.time()
    for epoch in tqdm(range(args.n_epochs), total = args.n_epochs):
        model.train()
        for i, slc in train_loader.iter_batch():
            feature, label, market_value , stock_index, _ = train_loader.get(slc)

            if args.model_name == 'HIST':
                pred = model(feature, stock2concept_matrix[stock_index], market_value)
            elif args.model_name == 'RSR':
                pred = model(feature, stock2stock_matrix[stock_index][:, stock_index])
            else:
                pred = model(feature)

            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
            optimizer.step()
        params_ckpt = copy.deepcopy(model.state_dict())
        params_list.append(params_ckpt)
        avg_params = average_params(params_list)
        # when evaluating, use the avg_params in the current time to evaluate
        model.load_state_dict(avg_params)
        # pprint("start validation")
        val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic = valid_epoch( model,
                                                                                          valid_loader,
                                                                                          args, stock2concept_matrix,
                                                                                          stock2stock_matrix,
                                                                                          )
        if val_score > best_score:
            best_param = copy.deepcopy(avg_params)
        else:
            # the model performance is not increasing
            stop_round += 1
            if stop_round >= args.early_stop:
                pprint('early stop')
                break
    t2 = time.time()
    # print("basic time:{}".format(t2 - t1))
    model.load_state_dict(best_param)



def set_seed(seed):
    np.random.seed(seed)    # numpy
    torch.manual_seed(seed) # cpu 为CPU设置种子用于生成随机数，以使得结果是确定的


def write(record, args):
    suffix = "increment%s_%s_%s_%s"%(args.incremental_number_days, args.baseline_start_date, args.incremental_start_date,
                                     args.baseline_end_date)
    df = pd.DataFrame({'time':record['time'], 'loss':record['loss'], 'score':record['score'], 'ic':record['ic'], 'rank_ic':record['rank_ic']})
    df.to_excel('./excel/' + args.model_name + '/' + suffix + '.xlsx')


def func(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def plot_norm(num_bins, losses):
    mpl.use('TkAgg')
    losses = np.array(losses)
    # n, bins, patches = plt.hist(x, 20, density=1, facecolor='blue', alpha=0.75)  #第二个参数是直方图柱子的数量
    mu = np.mean(losses)  # 计算均值
    sigma = np.std(losses)
    return mu, sigma




if __name__ == '__main__':
    """
    for prediction, maybe repeat 5, early_stop 10 is enough
    """
    args = parse_args()
    output_path = args.outdir
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not args.overwrite and os.path.exists(output_path+'/'+'info.json'):
        print('already runned, exit.')
        exit()
    pprint('create loaders...')

    start_time = datetime.datetime.strptime(args.train_start_date, '%Y-%m-%d')
    train_end_time = datetime.datetime.strptime(args.train_end_date, '%Y-%m-%d')
    end_time = datetime.datetime.strptime(args.test_end_date, '%Y-%m-%d')


    hanlder = {'class': 'Alpha360', 'module_path': 'qlib.contrib.data.handler', 'kwargs': {'start_time': start_time, 'end_time': end_time, 'fit_start_time': start_time, 'fit_end_time': train_end_time, 'instruments': args.data_set, 'infer_processors': [{'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True}}, {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}], 'learn_processors': [{'class': 'DropnaLabel'}, {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}}], 'label': ['Ref($close, -1) / $close - 1']}}
    segments = {'train': (args.train_start_date, args.train_end_date), 'baseline': (args.baseline_start_date, args.baseline_end_date), 'test': (args.test_start_date, args.test_end_date),
               'incre': (args.incremental_start_date, args.incremental_end_date), 'valid':(args.val_start_date, args.val_end_date)}
    # get dataset from qlib
    dataset = DatasetH(hanlder,segments)

    df_train, df_baseline, df_test, df_incre, df_valid = dataset.prepare(["train", "baseline", "test", "incre",'valid'], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L,)


    import pickle
    with open(args.market_value_path, "rb") as fh:
        # load market value
        df_market_value = pickle.load(fh)
    # df_market_value = pd.read_pickle(args.market_value_path)
    df_market_value = df_market_value/1000000000
    # market value of every day from 07 to 20
    stock_index = np.load(args.stock_index, allow_pickle=True).item()

    start_index = 0
    slc = slice(pd.Timestamp(args.train_start_date), pd.Timestamp(args.train_end_date))
    df_train['market_value'] = df_market_value[slc]
    df_train['market_value'] = df_train['market_value'].fillna(df_train['market_value'].mean())
    df_train['stock_index'] = 733
    df_train['stock_index'] = df_train.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    train_loader = DataLoader(df_train["feature"], df_train["label"], df_train['market_value'], df_train['stock_index'], batch_size=args.batch_size, pin_memory=args.pin_memory, start_index=start_index, device = device)
    # print(df_train)
    slc = slice(pd.Timestamp(args.baseline_start_date), pd.Timestamp(args.baseline_end_date))
    df_baseline['market_value'] = df_market_value[slc]
    df_baseline['market_value'] = df_baseline['market_value'].fillna(df_baseline['market_value'].mean())
    df_baseline['stock_index'] = 733
    df_baseline['stock_index'] = df_baseline.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    start_index += len(df_baseline.groupby(level=0).size())
    baseline_loader = DataLoader(df_baseline["feature"], df_baseline["label"], df_baseline['market_value'], df_baseline['stock_index'], batch_size=args.batch_size,
                              pin_memory=True, start_index=start_index, device=device)
    # print(df_baseline)
    slc = slice(pd.Timestamp(args.val_start_date), pd.Timestamp(args.val_end_date))
    df_valid['market_value'] = df_market_value[slc]
    df_valid['market_value'] = df_valid['market_value'].fillna(df_valid['market_value'].mean())
    df_valid['stock_index'] = 733
    df_valid['stock_index'] = df_valid.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    start_index += len(df_valid.groupby(level=0).size())
    valid_loader = DataLoader(df_valid["feature"], df_valid["label"], df_valid['market_value'], df_valid['stock_index'], batch_size=args.batch_size,
                              pin_memory=True, start_index=start_index, device=device)
    # print(df_valid)
    slc = slice(pd.Timestamp(args.test_start_date), pd.Timestamp(args.test_end_date))
    df_test['market_value'] = df_market_value[slc]
    df_test['market_value'] = df_test['market_value'].fillna(df_test['market_value'].mean())
    df_test['stock_index'] = 733
    df_test['stock_index'] = df_test.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    start_index += len(df_test.groupby(level=0).size())
    test_loader = DataLoader(df_test["feature"], df_test["label"], df_test['market_value'], df_test['stock_index'], pin_memory=True, start_index=start_index, device = device, batch_size=args.batch_size)
    # print(df_test)
    slc = slice(pd.Timestamp(args.incremental_start_date), pd.Timestamp(args.incremental_end_date))
    df_incre['market_value'] = df_market_value[slc]
    df_incre['market_value'] = df_incre['market_value'].fillna(df_incre['market_value'].mean())
    df_incre['stock_index'] = 733
    df_incre['stock_index'] = df_incre.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    start_index += len(df_incre.groupby(level=0).size())
    incremental_loader = DataLoader(df_incre["feature"], df_incre["label"], df_incre['market_value'], df_incre['stock_index'],
                              pin_memory=True, start_index=start_index, device=device, incremental_number_days = args.incremental_number_days)
    # print(df_incre)
    stock2concept_matrix = np.load(args.stock2concept_matrix)
    stock2stock_matrix = np.load(args.stock2stock_matrix)
    if args.model_name == 'HIST':
        # HIST need stock2concept matrix, send it to device
        stock2concept_matrix = torch.Tensor(stock2concept_matrix).to(device)
    if args.model_name == 'RSR':
        stock2stock_matrix = torch.Tensor(stock2stock_matrix).to(device)
        num_relation = stock2stock_matrix.shape[2]


    all_precision = []
    all_recall = []
    all_ic = []
    all_rank_ic = []


    print('create baseline model...')
    if args.model_name == 'SFM':
        model_baseline = get_model(args.model_name)(d_feat=args.d_feat, output_dim=32, freq_dim=25, hidden_size=args.hidden_size,
                                           dropout_W=0.5, dropout_U=0.5, device=device)
    elif args.model_name == 'ALSTM':
        # torch.backends.cudnn.enabled = False
        model_baseline = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, args.dropout, 'LSTM')
    elif args.model_name == 'Transformer':
        model_baseline = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, dropout=0.5)
    elif args.model_name == 'HIST':
        # torch.backends.cudnn.enabled = False
        model_baseline = get_model(args.model_name)(d_feat=args.d_feat, num_layers=args.num_layers, K=args.K)
    elif args.model_name == 'RSR':
        # torch.backends.cudnn.enabled = False
        model_baseline = get_model(args.model_name)(num_relation=num_relation, d_feat=args.d_feat, num_layers=args.num_layers)
    elif args.model_name == 'SFM':
        # torch.backends.cudnn.enabled = False
        model_baseline = get_model(args.model_name)(d_feat=args.d_feat, num_layers=args.num_layers)
    else:
        model_baseline = get_model(args.model_name)(d_feat=args.d_feat, num_layers=args.num_layers)

    # model_baseline = MLP(d_feat=args.d_feat, num_layers=args.num_layers)
    model_baseline.to(device)
    model_baseline.apply(weight_init)
    # print(model_baseline.state_dict())
    loss_list = []
    if not args.baseline_exist:
        optimizer = optim.Adam(model_baseline.parameters(), lr=args.lr)  # Learning rate
        best_score = -np.inf
        best_epoch = 0
        stop_round = 0
        # save best parameters
        best_param = copy.deepcopy(model_baseline.state_dict())
        params_list = collections.deque(maxlen=args.smooth_steps)

        t1 = time.time()
        for epoch in tqdm(range(args.n_epochs), total = args.n_epochs):
            model_baseline.train()
            # training(shuffle)

            for i, slc in baseline_loader.iter_batch():
                # print(i,slc)
                feature, label, market_value, stock_index, _ = baseline_loader.get(slc)

                # MLP model only use feature as input
                if args.model_name == 'HIST':
                    pred = model_baseline(feature, stock2concept_matrix[stock_index], market_value)
                elif args.model_name == 'RSR':
                    pred = model_baseline(feature, stock2stock_matrix[stock_index][:, stock_index])
                else:
                    pred = model_baseline(feature)

                loss = loss_fn(pred, label)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(model_baseline.parameters(), 3.)
                optimizer.step()

            params_ckpt = copy.deepcopy(model_baseline.state_dict())
            params_list.append(params_ckpt)
            avg_params = average_params(params_list)
            # when evaluating, use the avg_params in the current time to evaluate
            model_baseline.load_state_dict(avg_params)
            # pprint("start validation")
            val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic = valid_epoch(model_baseline, valid_loader,
                                                                                              args, stock2concept_matrix,
                                                                                             stock2stock_matrix)
            if val_score > best_score:
                # the model performance is increasing
                best_score = val_score
                stop_round = 0
                best_epoch = epoch
                best_param = copy.deepcopy(avg_params)
            else:
                # the model performance is not increasing
                stop_round += 1
                if stop_round >= args.early_stop:
                    pprint('early stop')
                    break

        t2 = time.time()
        baseline_time = t2 - t1
        print("baseline time:{}".format(baseline_time))
        model_baseline.load_state_dict(best_param)

        # save the model
        torch.save(model_baseline.state_dict(), output_path + '/model_baseline' + str(args.n_epochs) + str(args.incre_n_epochs) + '.pt')
    else:
        # load the model
        print("baseline loading")
        model_baseline.load_state_dict(torch.load(output_path + '/model_baseline' + str(args.n_epochs) + str(args.incre_n_epochs)+ '.pt'))

    baseline_loss, baseline_score, baseline_precision, baseline_recall, baseline_ic, baseline_rank_ic, losses = test_epoch(model_baseline,test_loader,
                                                                                                 args,stock2concept_matrix,stock2stock_matrix,
                                                                                                 prefix='Test')
    mu, sigma = plot_norm(num_bins=40, losses=losses)
    print("baseline model loss:{}, score:{}, ic:{}, rank_ic:{}".format(baseline_loss,baseline_score, baseline_ic, baseline_rank_ic))
    pprint("Baseline model finish")



    print("Basic model...")
    if args.model_name == 'SFM':
        model_train = get_model(args.model_name)(d_feat=args.d_feat, output_dim=32, freq_dim=25, hidden_size=args.hidden_size,
                                           dropout_W=0.5, dropout_U=0.5, device=device)
    elif args.model_name == 'ALSTM':
        model_train = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, args.dropout, 'LSTM')
    elif args.model_name == 'Transformer':
        model_train = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, dropout=0.5)
    elif args.model_name == 'HIST':
        model_train = get_model(args.model_name)(d_feat=args.d_feat, num_layers=args.num_layers, K=args.K)
    elif args.model_name == 'RSR':
        model_train = get_model(args.model_name)(num_relation=num_relation, d_feat=args.d_feat, num_layers=args.num_layers)

    else:
        model_train = get_model(args.model_name)(d_feat=args.d_feat, num_layers=args.num_layers)
    model_train.to(device)
    model_train.apply(weight_init)

    if not args.basic_exist:
        optimizer = optim.Adam(model_train.parameters(), lr=args.lr)
        best_score = -np.inf
        best_epoch = 0
        stop_round = 0
        # save best parameters
        best_param = copy.deepcopy(model_train.state_dict())
        params_list = collections.deque(maxlen=args.smooth_steps)

        train_epoch(model_train, optimizer, train_loader, args, stock2concept_matrix, stock2stock_matrix)
        # save the model
        torch.save(model_train.state_dict(), output_path + '/model_basic' + str(args.n_epochs) + str(args.incre_n_epochs)+ '.pt')
    else:
        # load the model
        print("basic loading")
        model_train.load_state_dict(torch.load(output_path + '/model_basic' + str(args.n_epochs) + str(args.incre_n_epochs) + '.pt'))



    # compute the loss, score, pre, recall, ic, rank_ic on train, valid and test data
    prev_loss, prev_score, prev_precision, prev_recall, prev_ic, prev_rank_ic, losses = test_epoch(model_train,
                                                                                                   test_loader, args,
                                                                                                   stock2concept_matrix,
                                                                                                   stock2stock_matrix,
                                                                                                   prefix='Train')



    print("increment model...")
    torch.cuda.empty_cache()
    prev_model = copy.deepcopy(model_train)
    if not args.incremental_exist:
        # record = incremental_epoch(model_train, prev_model, incremental_loader, args, mu = mu, sigma = sigma, alpha = 1)
        # write(record, args)
        increment_time = incremental_epoch(model_train, prev_model, incremental_loader, args, stock2concept_matrix = stock2concept_matrix, stock2stock_matrix = stock2stock_matrix, p = 10, mu = mu, sigma = sigma, alpha = 0.1)
        print("incremental model saving")
        torch.save(model_train.state_dict(), output_path + '/model_incre' + str(args.incre_n_epochs) + str(args.incremental_number_days) + '.pt')

    else:
        print("incremental model loading")
        model_train.load_state_dict(torch.load(output_path + '/model_incre' + str(args.incre_n_epochs) + str(args.incremental_number_days) + '.pt'))


    increment_loss, increment_score, increment_precision, increment_recall, increment_ic, increment_rank_ic, losses = test_epoch(model_train, test_loader,
                                                                                     args, stock2concept_matrix,stock2stock_matrix, prefix='Test')
    pprint('increment_loss %.6f, increment_score %.6f, increment_ic %.6f, increment_rank_ic %.6f' % (increment_loss, increment_score, increment_ic, increment_rank_ic))

    pprint('increment_ic精度损失为 %.6f %%, increment_rank_ic精度损失为 %.6f %%' % ((baseline_ic - increment_ic) * 100 /baseline_ic, (baseline_rank_ic - increment_rank_ic )* 100/baseline_rank_ic))
    pprint('增量更新的模型训练速度提升 %.6f %%  ' % ((baseline_time-increment_time) * 100 /baseline_time)  )





