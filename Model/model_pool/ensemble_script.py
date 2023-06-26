"""
this file is for some basic ensemble experiments on CSI 300
"""
import datetime
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.config import REG_US, REG_CN
import qlib
from utils.dataloader import DataLoader
import numpy as np
import pandas as pd
import torch
import argparse
from tqdm import tqdm
from models.model import MLP, HIST, GRU, LSTM, GAT, ALSTM, SFM, RSR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import json

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
# provider_uri = "../qlib_data/cn_data"  # target_dir
# provider_uri = "../qlib_data/cn_data_build" # our new dir
provider_uri = "~/.qlib/qlib_data/cn_data_build" # our new dir
qlib.init(provider_uri=provider_uri, region=REG_CN)


def metric_fn(preds, score='score'):
    preds = preds[~np.isnan(preds['label'])]
    precision = {}
    recall = {}
    temp = preds.groupby(level='datetime').apply(lambda x: x.sort_values(by=score, ascending=False))
    if len(temp.index[0]) > 2:
        temp = temp.reset_index(level=0).drop('datetime', axis=1)

    for k in [1, 3, 5, 10, 20, 30, 50, 100]:
        precision[k] = temp.groupby(level='datetime').apply(lambda x: (x.label[:k] > 0).sum() / k).mean()
        recall[k] = temp.groupby(level='datetime').apply(lambda x: (x.label[:k] > 0).sum() / (x.label > 0).sum()).mean()

    # mse = mean_squared_error(preds[['label']].values.tolist(),preds[[score]].values.tolist())
    ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x[score])).mean()
    rank_ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x[score], method='spearman')).mean()

    return precision, recall, ic, rank_ic


def get_model(model_name):
    if model_name.upper() == 'MLP':
        return MLP

    if model_name.upper() == 'LSTM':
        return LSTM

    if model_name.upper() == 'GRU':
        return GRU

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


def create_test_loaders(args, for_individual=False):
    """
    load qlib alpha360 data and split into train, validation and test loader
    if for individual model build dataloader, use blend_start/end_date replace test_start/end_date
    :param args:
    :return:
    """
    if for_individual:
        start_time = datetime.datetime.strptime(args.blend_start_date, '%Y-%m-%d')
        end_time = datetime.datetime.strptime(args.blend_end_date, '%Y-%m-%d')
        start_date = args.blend_start_date
        end_date = args.blend_end_date
    else:
        start_time = datetime.datetime.strptime(args.test_start_date, '%Y-%m-%d')
        end_time = datetime.datetime.strptime(args.test_end_date, '%Y-%m-%d')
        start_date = args.test_start_date
        end_date = args.test_end_date
    # 此处fit_start_time参照官方文档和代码
    hanlder = {'class': 'Alpha360', 'module_path': 'qlib.contrib.data.handler',
               'kwargs': {'start_time': start_time, 'end_time': end_time, 'fit_start_time': start_time,
                          'fit_end_time': end_time, 'instruments': args.data_set, 'infer_processors': [
                       {'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True}},
                       {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}],
                          'learn_processors': [{'class': 'DropnaLabel'},
                                               {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}}],
                          'label': ['Ref($close, -1) / $close - 1']}}
    segments = {'test': (start_date, end_date)}
    # get dataset from qlib
    dataset = DatasetH(hanlder, segments)
    # prepare return a list of df, df_test is the first one
    df_test = dataset.prepare(["test"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L, )[0]
    pd.to_pickle(df_test, './data/alpha360.pkl')
    # ----------------------------------------
    import pickle5 as pickle
    with open(args.market_value_path, "rb") as fh:
        # load market value
        df_market_value = pickle.load(fh)
        # the df_market_value save
    df_market_value = df_market_value / 1000000000
    stock_index = np.load(args.stock_index, allow_pickle=True).item()
    # stock_index is a dict and stock is the key, index is the value
    start_index = 0

    slc = slice(pd.Timestamp(start_date), pd.Timestamp(end_date))
    df_test['market_value'] = df_market_value[slc]
    df_test['market_value'] = df_test['market_value'].fillna(df_test['market_value'].mean())
    df_test['stock_index'] = 733
    df_test['stock_index'] = df_test.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    start_index += len(df_test.groupby(level=0).size())

    test_loader = DataLoader(df_test["feature"], df_test["label"], df_test['market_value'], df_test['stock_index'],
                             pin_memory=True, start_index=start_index, device=device)

    return test_loader


def inference(model, data_loader, stock2concept_matrix=None, stock2stock_matrix=None, model_name=''):
    model.eval()

    preds = []
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):
        # 当日切片
        feature, label, market_value, stock_index, index = data_loader.get(slc)
        with torch.no_grad():
            if model_name == 'HIST':
                pred = model(feature, stock2concept_matrix[stock_index], market_value)
            elif model_name == 'RSR':
                pred = model(feature, stock2stock_matrix[stock_index][:, stock_index])
            else:
                pred = model(feature)
            preds.append(
                pd.DataFrame({model_name + '_score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))

    preds = pd.concat(preds, axis=0)
    return preds


def find_time_interval(tset, time, stock, df=None, lookback=5):
    """
    :param tset:  time set that contains valid time in df
    :param lookback: number of trading days we need from past
    :return: a df contians lookback days stock prediction result
    """
    # return a dataframe that contains the stock's preidction score in the lookback days
    # current day is not included
    index = tset.index(time)
    if index <= (lookback - 1):
        # no enough history data
        return None
    else:
        res_df=pd.DataFrame()
        while res_df.shape[0] < lookback and index > 0:
            index = index - 1
            if (tset[index], stock) in df.index:
                res_df = res_df.append(df.loc[tset[index], stock])
        if res_df.shape[0] != lookback:
            return None
        else:
            # have enough lookback data
            return res_df


def prediction(args, test_loader):
    """
    test single model first, load model from folder and do prediction
    :param args:
    :return:
    """
    # test_loader = create_test_loaders(args, for_individual=for_individual)
    stock2concept_matrix = args.stock2concept_matrix
    stock2stock_matrix = args.stock2stock_matrix
    print('load model ', args.model_name)
    if args.model_name == 'SFM':
        model = get_model(args.model_name)(d_feat=args.d_feat, output_dim=32, freq_dim=25, hidden_size=args.hidden_size,
                                           dropout_W=0.5, dropout_U=0.5, device=device)
    elif args.model_name == 'ALSTM':
        model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, args.dropout, 'LSTM')
    elif args.model_name == 'Transformer':
        model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, dropout=0.5)
    elif args.model_name == 'HIST':
        # HIST need stock2concept matrix, send it to device
        stock2concept_matrix = torch.Tensor(np.load(stock2concept_matrix)).to(device)
        model = get_model(args.model_name)(d_feat=args.d_feat, num_layers=args.num_layers, K=args.K)
    elif args.model_name == 'RSR':
        stock2stock_matrix = torch.Tensor(np.load(stock2stock_matrix)).to(device)
        num_relation = stock2stock_matrix.shape[2]  # the number of relations
        model = get_model(args.model_name)(num_relation=num_relation, d_feat=args.d_feat, num_layers=args.num_layers)
    else:
        model = get_model(args.model_name)(d_feat=args.d_feat, num_layers=args.num_layers)
    model.to(device)
    model.load_state_dict(torch.load(args.model_dir + '/model.bin', map_location=device))
    print('predict in ', args.model_name)
    pred = inference(model, test_loader, stock2concept_matrix, stock2stock_matrix, args.model_name)
    return pred


def average(model_pool, prefix):
    pred_group = []
    for m in model_pool:
        model_dir = prefix + m
        param_dict = json.load(open(model_dir + '/info.json'))['config']
        param_dict['model_dir'] = model_dir
        args = parse_args(param_dict)
        if m == 'HIST':
            args.stock2concept = param_dict['stock2concept_matrix']
            # args.market_value_path = param_dict['market_value_path']
        if m == 'RSR':
            args.stock2stock_matrix = param_dict['stock2stock_matrix']
        temp = prediction(args)
        pred_group.append(temp)
    res = pd.concat(pred_group, axis=1)
    res = res.loc[:, ~res.columns.duplicated()].copy()
    res['average_score'] = (res['GATs_score']+res['GRU_score']+res['LSTM_score']+res['MLP_score'])/4
    report = dict()
    for name in ['GATs_score', 'GRU_score', 'LSTM_score','MLP_score','average_score']:
        precision, recall, ic, rank_ic, mse = metric_fn(res, score=name)
        print(('%s: IC %.6f Rank IC %.6f') % (name, ic, rank_ic))
        report[name + '-Precision'] = precision
        report[name+ '-Recall'] = recall
        report[name + '-IC'] = ic
        # res[name+'-ICIR'] = ic.mean() / ic.std()
        report[name + '-RankIC'] = rank_ic
    json.dump(report, open('ensemble_output/ave_report.json', 'w'), indent=6)
    # save test report
    res.to_pickle('ensemble_output/average_model_res.pkl')
    # save original dataframe file


def blending(model_pool, prefix):
    pred_group = []
    temp_p = json.load(open(prefix+model_pool[0] + '/info.json'))['config']
    temp_p['model_dir'] = prefix+model_pool[0]
    # temp param dict to load data loader
    args = parse_args(temp_p)
    test_loader = create_test_loaders(args, for_individual=True)
    # for individual = True, the dataloader load validation dataset
    for m in model_pool:
        model_dir = prefix + m
        param_dict = json.load(open(model_dir + '/info.json'))['config']
        param_dict['model_dir'] = model_dir
        args = parse_args(param_dict)
        if m == 'HIST':
            args.stock2concept = param_dict['stock2concept_matrix']
            # args.market_value_path = param_dict['market_value_path']
        if m == 'RSR':
            args.stock2stock_matrix = param_dict['stock2stock_matrix']
        temp = prediction(args, test_loader)
        pred_group.append(temp)
    data = pd.concat(pred_group, axis=1)
    data = data.loc[:, ~data.columns.duplicated()].copy()
    score_pool = [x+'_score' for x in model_pool]
    X = data[score_pool].values.tolist()
    y = data[['label']].values.tolist()
    reg = LinearRegression().fit(X, y)
    pred_group = []
    test_loader = create_test_loaders(args)
    # for individual = False
    for m in model_pool:
        # train on test data
        model_dir = prefix + m
        param_dict = json.load(open(model_dir + '/info.json'))['config']
        param_dict['model_dir'] = model_dir
        args = parse_args(param_dict)
        if m == 'HIST':
            args.stock2concept = param_dict['stock2concept_matrix']
            # args.market_value_path = param_dict['market_value_path']
        if m == 'RSR':
            args.stock2stock_matrix = param_dict['stock2stock_matrix']
        temp = prediction(args, test_loader)
        pred_group.append(temp)
    data = pd.concat(pred_group, axis=1)
    data = data.loc[:, ~data.columns.duplicated()].copy()
    X_t = data[score_pool].values.tolist()
    y_pred = reg.predict(X_t)
    data['blend_score'] = y_pred
    data['average_score'] = sum(data[x] for x in score_pool)/len(score_pool)
    report = dict()
    for name in score_pool+['average_score', 'blend_score']:
        precision, recall, ic, rank_ic = metric_fn(data, score=name)
        print(('%s: IC %.6f Rank IC %.6f') % (name, ic, rank_ic))
        report[name + '-Precision'] = precision
        report[name+ '-Recall'] = recall
        report[name + '-IC'] = ic
        # res[name+'-ICIR'] = ic.mean() / ic.std()
        report[name + '-RankIC'] = rank_ic
        # report[name+'-mse'] = mse
    json.dump(report, open('ensemble_output/blend_report_22.json', 'w'), indent=6)
    data.to_pickle('ensemble_output/blend_model_res_22.pkl')


def build_level2_dataset(model_pool, prefix):
    pred_group = []
    temp_p = json.load(open(prefix+model_pool[0] + '/info.json'))['config']
    temp_p['model_dir'] = prefix+model_pool[0]
    # temp param dict to load data loader
    args = parse_args(temp_p)
    test_loader = create_test_loaders(args)
    # return 0
    for m in model_pool:
        model_dir = prefix + m
        param_dict = json.load(open(model_dir + '/info.json'))['config']
        param_dict['model_dir'] = model_dir
        args = parse_args(param_dict)
        if m == 'HIST':
            args.stock2concept_matrix = param_dict['stock2concept_matrix']
        if m == 'RSR':
            args.stock2stock_matrix = param_dict['stock2stock_matrix']
        temp = prediction(args, test_loader)
        pred_group.append(temp)
    data = pd.concat(pred_group, axis=1)
    data = data.loc[:, ~data.columns.duplicated()].copy()
    # data.to_pickle('ensemble_output/level2_dataset_8m_newmkfile.pkl')


def temp_model(model_pool):
    # use data from previous experiments
    data = pd.read_pickle('ensemble_output/blend_model_res.pkl')
    score_pool = [x + '_score' for x in model_pool]

    def train_a_temp_lr_learner(score_pool, data, lookback=10):
        # this function is used to train a temp learner, that could generate a blend score for one day
        # input should be score_pool, data, lookback
        # output is a dataframe with new score from the temp learner
        timeset = list(dict.fromkeys([x[0] for x in data.index.drop_duplicates().values.tolist()]))
        stockset = [x[1] for x in data.index.drop_duplicates().values.tolist()]
        for time in tqdm(timeset):
            for stock in stockset:
                if (time, stock) in data.index:
                    # the combo have value in data
                    temp = find_time_interval(timeset, time, stock, data, lookback)
                    if temp is not None:
                        x = temp[score_pool].values.tolist()
                        y = temp[['label']].values.tolist()
                        reg = LinearRegression().fit(x, y)
                        x_t = np.array(data.loc[time, stock][score_pool].values.tolist()).reshape(1, -1)
                        y_t = reg.predict(x_t)[0][0]
                        data.loc[(time, stock), 'dyb_score'] = y_t
        return data

    res = train_a_temp_lr_learner(score_pool, data)
    res.to_pickle('ensemble_output/dyb_model_res_lookback_10.pkl')
    report = dict()
    for name in score_pool + ['average_score', 'blend_score', 'dyb_score']:
        precision, recall, ic, rank_ic = metric_fn(data, score=name)
        print(('%s: IC %.6f Rank IC %.6f') % (name, ic, rank_ic))
        report[name + '-Precision'] = precision
        report[name + '-Recall'] = recall
        report[name + '-IC'] = ic
        # res[name+'-ICIR'] = ic.mean() / ic.std()
        report[name + '-RankIC'] = rank_ic
        # report[name + 'mse'] = mse
    json.dump(report, open('ensemble_output/dyb_report_lookback_10.json', 'w'), indent=6)


def parse_args(param_dict):
    """
    deliver arguments from json file to program
    :param param_dict: a dict that contains model info
    :return: no return
    """
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model_name', default=param_dict['model_name'])
    parser.add_argument('--d_feat', type=int, default=param_dict['d_feat'])
    parser.add_argument('--hidden_size', type=int, default=param_dict['hidden_size'])
    parser.add_argument('--num_layers', type=int, default=param_dict['num_layers'])
    parser.add_argument('--dropout', type=float, default=param_dict['dropout'])
    parser.add_argument('--K', type=int, default=param_dict['K'])
    # data
    parser.add_argument('--data_set', type=str, default=param_dict['data_set'])
    parser.add_argument('--pin_memory', action='store_false', default=param_dict['pin_memory'])
    parser.add_argument('--batch_size', type=int, default=param_dict['batch_size'])  # -1 indicate daily batch
    parser.add_argument('--least_samples_num', type=float, default=param_dict['least_samples_num'])
    parser.add_argument('--label', default='')  # specify other labels
    parser.add_argument('--blend_start_date', default='2015-01-01')
    parser.add_argument('--blend_end_date', default='2017-12-31')
    parser.add_argument('--test_start_date', default='2007-01-01')
    parser.add_argument('--test_end_date', default='2022-12-31')
    # parser.add_argument('--market_value_path', default=param_dict['market_value_path'])
    # 强制使用07to22的市值文件
    parser.add_argument('--market_value_path', default='./data/csi300_market_value_07to22.pkl')
    parser.add_argument('--stock2concept_matrix', default=None)
    parser.add_argument('--stock2stock_matrix', default=None)
    parser.add_argument('--stock_index', default=param_dict['stock_index'])
    parser.add_argument('--model_dir', default=param_dict['model_dir'])
    parser.add_argument('--dir_prefix', default='./ouput/csi_', help='the prefix of model dict')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    #     param_dict = json.load(open('./output/csi300_GRU/info.json'))['config']
    #     param_dict['model_dir'] = './output/csi300_GRU'
    #     args = parse_args(param_dict)
    #     print(prediction(args))
    #     main(['GATs', 'GRU', 'LSTM', 'MLP'], './output/csi300_')
    # model_pool = ['GATs', 'GRU', 'LSTM', 'MLP', 'ALSTM', 'SFM', 'HIST','RSR']
    model_pool = ['MLP']
    # blending(model_pool, './output/csi300_')
    # temp_model(model_pool)
    build_level2_dataset(model_pool, './output/csi300_') #2015-2022全收录
