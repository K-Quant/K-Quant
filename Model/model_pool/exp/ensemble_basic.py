import datetime
import sys
sys.path.insert(0, sys.path[0]+"/../")
from utils.dataloader import create_test_loaders
from utils.utils import DotDict
import numpy as np
import pandas as pd
import torch
import argparse
import pickle
from tqdm import tqdm
from models.model import MLP, HIST, GRU, LSTM, GAT, ALSTM, SFM, RSR, relation_GATs, relation_GATs_3heads, KEnhance
from models.ensemble_model import ReweightModel, PerfomanceBasedModel
from qlib.contrib.model.pytorch_transformer import Transformer
from models.DLinear import DLinear_model
from models.Autoformer import Model as autoformer
from models.Crossformer import Model as crossformer
from models.ETSformer import Model as ETSformer
from models.FEDformer import Model as FEDformer
from models.FiLM import Model as FiLM
from models.Informer import Model as Informer
from models.PatchTST import Model as PatchTST
import json
from sklearn.linear_model import LinearRegression

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
    'relation_GATs_3heads',
    'KEnhance'
]


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

    if model_name.upper() == 'NRSR':
        return RSR

    if model_name.upper() == 'PATCHTST':
        return PatchTST

    if model_name.upper() == 'KENHANCE':
        return KEnhance



    raise ValueError('unknown model name `%s`' % model_name)


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
    icir = ic/preds.groupby(level='datetime').apply(lambda x: x.label.corr(x[score])).std()
    rank_icir = rank_ic/preds.groupby(level='datetime').apply(lambda x: x.label.corr(x[score], method='spearman')).std()

    return precision, recall, ic, rank_ic, icir, rank_icir


def inference(model, data_loader, stock2concept_matrix=None, stock2stock_matrix=None, model_name='', inference_name=''):
    model.eval()
    preds = []
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):
        # 当日切片
        feature, label, market_value, stock_index, index, mask = data_loader.get(slc)
        # feature, label, index = data_loader.get(slc)
        with torch.no_grad():
            if model_name == 'HIST':
                pred = model(feature, stock2concept_matrix[stock_index], market_value)
            elif model_name in relation_model_dict:
                pred = model(feature, stock2stock_matrix[stock_index][:, stock_index])
            elif model_name in time_series_library:
                pred = model(feature, mask)
            else:
                pred = model(feature)
            preds.append(
                pd.DataFrame({inference_name + '_score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))

    preds = pd.concat(preds, axis=0)
    return preds


def _prediction(param_dict, test_loader, device):
    """
    test single model first, load model from folder and do prediction
    """
    # test_loader = create_test_loaders(args, for_individual=for_individual)
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
    elif param_dict['model_name'] in time_series_library:
        model = get_model(param_dict['model_name'])(DotDict(param_dict))
    else:
        model = get_model(param_dict['model_name'])(d_feat=param_dict['d_feat'], num_layers=param_dict['num_layers'])
    model.to(device)
    if param_dict['incre_model_path']:
        model.load_state_dict(torch.load(param_dict['incre_model_path'] + '/model.bin', map_location=device))
        print('predict in ', param_dict['model_name'], ' incremental model')
    else:
        model.load_state_dict(torch.load(param_dict['model_dir'] + '/model.bin', map_location=device))
        print('predict in ', param_dict['model_name'])
    # pred = inference(model, test_loader, stock2concept_matrix, stock2stock_matrix, param_dict['model_name'])
    pred = inference(model, test_loader, stock2concept_matrix, stock2stock_matrix,
                     param_dict['model_name'], param_dict['inference_name'])
    return pred


def batch_prediction(args, model_pool, device):
    initial_param = json.load(open(args.prefix+model_pool[0]+'/info.json'))['config']
    test_loader = create_test_loaders(args, initial_param, device=device)
    output_group = []
    for i in model_pool:
        model_path = args.prefix+i
        param_dict = json.load(open(model_path+'/info.json'))['config']
        param_dict['model_dir'] = model_path
        param_dict['inference_name'] = i  # create a inference name to avoid model name is different with name in model pool like RSR and RSR_hidy

        if args.incremental_mode:
            param_dict['incre_model_path'] = args.incre_prefix+i+'_incre'
        else:
            param_dict['incre_model_path'] = None
        pred = _prediction(param_dict, test_loader, device)
        output_group.append(pred)

    data = pd.concat(output_group, axis=1)
    data = data.loc[:, ~data.columns.duplicated()].copy()
    return data


def average_and_blend(args, data, model_pool):
    model_score = [i+'_score' for i in model_pool]
    data['average_score'] = data[model_score].mean(axis=1)
    # for blend, we need to split it to train set and test set, run linear regression to learn weight
    # and test the performance on test set
    btrain_slc = slice(pd.Timestamp(args.test_start_date), pd.Timestamp(args.blend_split_date))
    btest_slc = slice(pd.Timestamp(args.blend_split_date), pd.Timestamp(args.test_end_date))
    btrain_data = data[btrain_slc]
    btest_data = data[btest_slc]
    # --------train a linear model to learn the weight of each model--------------------
    # here we split the dataset and only evaluate on the test_data
    X = btrain_data[model_score].values.tolist()
    y = btrain_data[['label']].values.tolist()
    reg = LinearRegression().fit(X, y)
    X_t = btest_data[model_score].values.tolist()
    y_pred = reg.predict(X_t)
    btest_data['blend_score'] = y_pred
    report = pd.DataFrame()
    for name in model_score+['average_score', 'blend_score']:
        temp = dict()
        temp['model'] = name
        precision, recall, ic, rank_ic, icir, rank_icir = metric_fn(btest_data, score=name)
        temp['P@3'] = precision[3]
        temp['P@5'] = precision[5]
        temp['P@10'] = precision[10]
        temp['P@30'] = precision[30]
        temp['IC'] = ic
        temp['ICIR'] = icir
        temp['RankIC'] = rank_ic
        temp['RankICIR'] = rank_icir
        report = report.append(temp, ignore_index=True)
    return btest_data, report


def sjtu_ensemble(args, data, model_pool):
    np.random.seed(0)
    model_score = [i+'_score' for i in model_pool]
    btrain_slc = slice(pd.Timestamp(args.test_start_date), pd.Timestamp(args.blend_split_date))
    btest_slc = slice(pd.Timestamp(args.blend_split_date), pd.Timestamp(args.test_end_date))
    btrain_data = data[btrain_slc]
    btest_data = data[btest_slc]
    with open('output/ensemble_model/sjtumodel.pkl', 'rb') as f:
        ensemble_model = pickle.load(f)
    X = btrain_data[model_score].values
    y = btrain_data[['label']].values
    X_t = btest_data[model_score].values
    y_t = btest_data[['label']].values
    ensemble_model.train(X, y)
    y_pred_no_retrain = ensemble_model(X_t)
    y_pred_retrain = ensemble_model.predict(X, y, X_t, y_t,
                                            retrain_interval=300, max_retrain_samples=-1, progress_bar=True)
    btest_data['ensemble_retrain_score'] = y_pred_retrain
    btest_data['ensemble_no_retrain_score'] = y_pred_no_retrain
    pbm = PerfomanceBasedModel()
    pbm.train(X, y)
    btest_data['Perfomance_based_ensemble_score'] = pbm(X_t)
    report = pd.DataFrame()
    for name in model_score+['ensemble_retrain_score', 'ensemble_no_retrain_score', 'Perfomance_based_ensemble_score']:
        temp = dict()
        temp['model'] = name
        precision, recall, ic, rank_ic, icir, rank_icir = metric_fn(btest_data, score=name)
        temp['P@3'] = precision[3]
        temp['P@5'] = precision[5]
        temp['P@10'] = precision[10]
        temp['P@30'] = precision[30]
        temp['IC'] = ic
        temp['ICIR'] = icir
        temp['RankIC'] = rank_ic
        temp['RankICIR'] = rank_icir
        report = report.append(temp, ignore_index=True)
    return btest_data, report


def find_time_interval(tset, time, df=None, lookback=5):
    # return a dataframe that contains all stock's preidction score in the lookback days
    # current day is not included
    index = tset.index(time)
    if index <= (lookback - 1):
        # no enough history data
        res_df = pd.DataFrame()
        res_df = df.loc[tset[:index]]
        return res_df
    else:
        res_df=pd.DataFrame()
        time_interval = tset[index-lookback:index]
        res_df = df.loc[time_interval]
        return res_df


def sim_linear(data, model_pool, lookback=30, eva_type='ic', select_num=5):
    # this function is used to train a temp learner, that could generate a blend score for one day
    # input should be score_pool, data, lookback
    # output is a dataframe with new score from the temp learner
    # 通过对每个模型在之前交易日内的表现筛选模型通过线性回归组合
    score_pool = [i + '_score' for i in model_pool]
    timeset = list(dict.fromkeys([x[0] for x in data.index.drop_duplicates().values.tolist()]))
    for time in tqdm(timeset):
        temp = find_time_interval(timeset, time, data, lookback)
        if len(temp) > 0:
            report = {}
            for name in score_pool:
                if eva_type == 'ic+rank_ic':
                    precision, recall, ic, rank_ic, icir, rank_icir= metric_fn(temp, score=name)
                    report[name] = ic + rank_ic
                if eva_type == 'ic':
                    precision, recall, ic, rank_ic, icir, rank_icir= metric_fn(temp, score=name)
                    report[name] = ic
                    # evaluate on lookback data
            report = dict(sorted(report.items(), key=lambda item: item[1]))
            ordered_model_list = list(report.keys())
            # 经过大量训练后其实模型选择影响不大，绝大多数情况都会选择到某几个模型
            if 0 < select_num <= len(ordered_model_list):
                cur_pool = ordered_model_list[len(ordered_model_list)-select_num:]
            else:
                print('the chosen number of model is not valid, select all models dy default')
                cur_pool = ordered_model_list

            x = temp[cur_pool].values.tolist()
            y = temp[['label']].values.tolist()
            model = LinearRegression().fit(x,y)
            x_t = np.array(data.loc[time][cur_pool].values.tolist())
            y_t = model.predict(x_t)
            data.loc[time,'dynamic_ensemble_score']= y_t
    report = pd.DataFrame()
    for name in score_pool+['dynamic_ensemble_score']:
        temp = dict()
        temp['model'] = name
        precision, recall, ic, rank_ic, icir, rank_icir = metric_fn(data, score=name)
        temp['P@3'] = precision[3]
        temp['P@5'] = precision[5]
        temp['P@10'] = precision[10]
        temp['P@30'] = precision[30]
        temp['IC'] = ic
        temp['ICIR'] = icir
        temp['RankIC'] = rank_ic
        temp['RankICIR'] = rank_icir
        report = report.append(temp, ignore_index=True)
    return data.dropna(), report


def main(args, device):
    model_pool = ['GRU', 'LSTM', 'GATs', 'MLP', 'ALSTM', 'SFM']
    all_model_pool = ['RSR_hidy_is', 'KEnhance', 'LSTM', 'GRU', 'GATs', 'MLP', 'ALSTM', 'SFM', 'HIST']
    output = batch_prediction(args, model_pool, device)
    output, report = average_and_blend(args, output, all_model_pool)
    output, report = sjtu_ensemble(args, output, all_model_pool)
    output, report = sim_linear(output, all_model_pool)
    pd.to_pickle(output, 'pred_output/all_in_one_incre.pkl')
    print(output.head())
    # print(report)


def parse_args():
    """
    deliver arguments from json file to program
    :param param_dict: a dict that contains model info
    :return: no return
    """
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--test_start_date', default='2021-01-01')
    parser.add_argument('--blend_split_date', default='2021-06-01')
    parser.add_argument('--test_end_date', default='2023-06-30')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--incremental_mode', default=False, help='load incremental updated models or not')
    parser.add_argument('--prefix', default='output/for_platform/')
    parser.add_argument('--incre_prefix', default='output/for_platform/INCRE/')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    main(args, device)
