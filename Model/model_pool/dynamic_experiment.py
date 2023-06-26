import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from numpy import dot
from numpy.linalg import norm

"""
used to find best top K and best lookback days in ensemble experiment
"""


def metric_fn(preds, score='score'):
    # preds = preds[~np.isnan(preds['label'])]
    preds = preds[~np.isnan(preds[score])]
    precision = {}
    recall = {}
    temp = preds.groupby(level='datetime').apply(lambda x: x.sort_values(by=score, ascending=False))
    if len(temp.index[0]) > 2:
        temp = temp.reset_index(level=0).drop('datetime', axis=1)

    for k in [1, 3, 5, 10, 20, 30, 50, 100]:
        precision[k] = temp.groupby(level='datetime').apply(lambda x: (x.label[:k] > 0).sum() / k).mean()
        recall[k] = temp.groupby(level='datetime').apply(lambda x: (x.label[:k] > 0).sum() / (x.label > 0).sum()).mean()

    mse = mean_squared_error(preds[['label']].values.tolist(),preds[[score]].values.tolist())
    ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x[score])).mean()
    rank_ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x[score], method='spearman')).mean()

    return precision, recall, ic, rank_ic, mse


def find_time_interval(tset, time, df=None, lookback=5):
    # return a dataframe that contains all stock's preidction score in the lookback days
    # current day is not included
    index = tset.index(time)
    if index <= (lookback - 1):
        # no enough history data
        res_df = df.loc[timeset[:index]]
        return res_df
    else:
        time_interval = timeset[index-lookback:index]
        res_df = df.loc[time_interval]
        return res_df


def vote_linear(score_pool, data, lookback=90):
    # this function is used to train a temp learner, that could generate a blend score for one day
    # input should be score_pool, data, lookback
    # output is a dataframe with new score from the temp learner
    # 通过对每个模型在之前交易日内的表现筛选模型通过线性回归组合
    data_copy = data.copy(deep=True)
    timeset = list(dict.fromkeys([x[0] for x in data_copy.index.drop_duplicates().values.tolist()]))
    for time in timeset:
        temp = find_time_interval(timeset, time, data_copy, lookback)
        if len(temp) > 0:
            report = {}
            for name in score_pool:
                precision, recall, ic, rank_ic, mse = metric_fn(temp, score=name)
                score = ic + rank_ic + (precision[3]+precision[5]+precision[10]+precision[30])/40
                # score = ic + rank_ic
                report[name] = score
                # evaluate on previous data
            report = dict(sorted(report.items(), key=lambda item: item[1]))
            ordered_model_list = list(report.keys())
            # 经过大量训练后其实模型选择影响不大，绝大多数情况都会选择到某几个模型
            # keep top 3 models or 4
            cur_pool = ordered_model_list[3:]
            x = temp[cur_pool].values.tolist()
            y = temp[['label']].values.tolist()
            model = LinearRegression().fit(x,y)
            x_t = np.array(data.loc[time][cur_pool].values.tolist())
            y_t = model.predict(x_t)
            data_copy.loc[time, 'dyb_score'] = y_t
    return data_copy


def find_best_topk(score_pool, data, lookback=90):
    # this function is used to train a temp learner, that could generate a blend score for one day
    # input should be score_pool, data, lookback
    # output is a dataframe with new score from the temp learner
    # 通过对每个模型在之前交易日内的表现筛选模型通过线性回归组合
    timeset = list(dict.fromkeys([x[0] for x in data.index.drop_duplicates().values.tolist()]))
    for time in tqdm(timeset):
        temp = find_time_interval(timeset, time, data, lookback)
        if temp is not None:
            report = {}
            for name in score_pool:
                precision, recall, ic, rank_ic, mse = metric_fn(temp, score=name)
                score = ic + rank_ic + (precision[3]+precision[5]+precision[10]+precision[30])/40
                # score = ic + rank_ic
                report[name] = score
                # evaluate on previous data
            report = dict(sorted(report.items(), key=lambda item: item[1]))
            ordered_model_list = list(report.keys())
            for i in range(len(ordered_model_list)):
                cur_pool = ordered_model_list[i:]
                x = temp[cur_pool].values.tolist()
                y = temp[['label']].values.tolist()
                model = LinearRegression().fit(x,y)
                x_t = np.array(data.loc[time][cur_pool].values.tolist())
                y_t = model.predict(x_t)
                data.loc[time, 'dyb_score_top'+str(7-i)] = y_t
    df = pd.DataFrame()
    for i in range(7):
        temp = dict()
        precision, recall, ic, rank_ic, mse = metric_fn(data, score='dyb_score_top'+str(7-i))
        temp['model'] = 'dyb_score_top'+str(7-i)
        temp['P@3'] = precision[3]
        temp['P@5'] = precision[5]
        temp['P@10'] = precision[10]
        temp['P@30'] = precision[30]
        temp['IC'] = ic
        temp['RankIC'] = rank_ic
        temp['mse'] = mse
        df = df.append(temp, ignore_index=True)
    df.to_pickle('ensemble_output/find_best_topk.pkl')
    return None


def find_best_topk_sim_linear(score_pool, data, lookback=90):
    # this function is used to train a temp learner, that could generate a blend score for one day
    # input should be score_pool, data, lookback
    # output is a dataframe with new score from the temp learner
    timeset = list(dict.fromkeys([x[0] for x in data.index.drop_duplicates().values.tolist()]))
    for time in tqdm(timeset):
        temp = find_time_interval(timeset, time, data, lookback)
        if temp is not None:
            report = {}
            for name in score_pool:
                a = np.array(temp['label'].values.tolist())
                b = np.array(temp[name].values.tolist())
                report[name] = dot(a, b) / (norm(a) * norm(b))
            report = dict(sorted(report.items(), key=lambda item: item[1]))
            ordered_model_list = list(report.keys())
            for i in range(len(ordered_model_list)):
                cur_pool = ordered_model_list[i:]
                x = temp[cur_pool].values.tolist()
                y = temp[['label']].values.tolist()
                model = LinearRegression().fit(x,y)
                x_t = np.array(data.loc[time][cur_pool].values.tolist())
                y_t = model.predict(x_t)
                data.loc[time, 'dyb_score_top'+str(7-i)] = y_t
    df = pd.DataFrame()
    for i in range(7):
        temp = dict()
        precision, recall, ic, rank_ic, mse = metric_fn(data, score='dyb_score_top'+str(7-i))
        temp['model'] = 'dyb_score_top'+str(7-i)
        temp['P@3'] = precision[3]
        temp['P@5'] = precision[5]
        temp['P@10'] = precision[10]
        temp['P@30'] = precision[30]
        temp['IC'] = ic
        temp['RankIC'] = rank_ic
        temp['mse'] = mse
        df = df.append(temp, ignore_index=True)
    df.to_pickle('ensemble_output/find_best_topk.pkl')
    return None


def find_best_topk_eva_att(score_pool, data, lookback=90):
    # this function is used to train a temp learner, that could generate a blend score for one day
    # input should be score_pool, data, lookback
    # output is a dataframe with new score from the temp learner
    timeset = list(dict.fromkeys([x[0] for x in data.index.drop_duplicates().values.tolist()]))
    for time in tqdm(timeset):
        temp = find_time_interval(timeset, time, data, lookback)
        if temp is not None:
            report = {}
            for name in score_pool:
                a = np.array(temp['label'].values.tolist())
                b = np.array(temp[name].values.tolist())
                report[name] = dot(a, b) / (norm(a) * norm(b))
            report = dict(sorted(report.items(), key=lambda item: item[1]))
            ordered_model_list = list(report.keys())
            for i in range(len(ordered_model_list)):
                cur_pool = ordered_model_list[i:]
                model_score = list(report.values())[i:]
                model_weight = np.exp(model_score) / np.sum(np.exp(model_score), axis=0)
                x_t = np.array(data.loc[time][cur_pool].values.tolist())
                y_t = np.dot(x_t, np.transpose(model_weight))
                data.loc[time, 'dyb_score_top'+str(7-i)] = y_t
    df = pd.DataFrame()
    for i in range(7):
        temp = dict()
        precision, recall, ic, rank_ic, mse = metric_fn(data, score='dyb_score_top'+str(7-i))
        temp['model'] = 'dyb_score_top'+str(7-i)
        temp['P@3'] = precision[3]
        temp['P@5'] = precision[5]
        temp['P@10'] = precision[10]
        temp['P@30'] = precision[30]
        temp['IC'] = ic
        temp['RankIC'] = rank_ic
        temp['mse'] = mse
        df = df.append(temp, ignore_index=True)
    df.to_pickle('ensemble_output/find_best_topk.pkl')
    return None


def find_best_topk_sim_att(score_pool, data, lookback=90):
    # this function is used to train a temp learner, that could generate a blend score for one day
    # input should be score_pool, data, lookback
    # output is a dataframe with new score from the temp learner
    timeset = list(dict.fromkeys([x[0] for x in data.index.drop_duplicates().values.tolist()]))
    for time in tqdm(timeset):
        temp = find_time_interval(timeset, time, data, lookback)
        if temp is not None:
            report = {}
            for name in score_pool:
                precision, recall, ic, rank_ic, mse= metric_fn(temp, score=name)
                score = ic + rank_ic + (precision[3]+precision[5]+precision[10]+precision[30])/40
                # score = ic + rank_ic
                report[name] = score
                # evaluate on previous data
            report = dict(sorted(report.items(), key=lambda item: item[1]))
            ordered_model_list = list(report.keys())
            for i in range(len(ordered_model_list)):
                cur_pool = ordered_model_list[i:]
                model_score = list(report.values())[i:]
                model_weight = np.exp(model_score) / np.sum(np.exp(model_score), axis=0)
                x_t = np.array(data.loc[time][cur_pool].values.tolist())
                y_t = np.dot(x_t, np.transpose(model_weight))
                data.loc[time, 'dyb_score_top'+str(7-i)] = y_t
    df = pd.DataFrame()
    for i in range(7):
        temp = dict()
        precision, recall, ic, rank_ic, mse = metric_fn(data, score='dyb_score_top'+str(7-i))
        temp['model'] = 'dyb_score_top'+str(7-i)
        temp['P@3'] = precision[3]
        temp['P@5'] = precision[5]
        temp['P@10'] = precision[10]
        temp['P@30'] = precision[30]
        temp['IC'] = ic
        temp['RankIC'] = rank_ic
        temp['mse'] = mse
        df = df.append(temp, ignore_index=True)
    df.to_pickle('ensemble_output/find_best_topk.pkl')
    return None


data = pd.read_pickle('ensemble_output/blend_model_res.pkl')
score_pool=['GATs_score','GRU_score','LSTM_score','MLP_score','ALSTM_score','SFM_score','HIST_score']
timeset = list(dict.fromkeys([x[0] for x in data.index.drop_duplicates().values.tolist()]))
stockset = [x[1] for x in data.index.drop_duplicates().values.tolist()]
# res = vote_linear(score_pool, data, lookback=90)
# print(metric_fn(res, score='dyb_score'))
# find_best_topk(score_pool,data)
df = pd.DataFrame()
for back in tqdm(range(7, 120)):
    res = vote_linear(score_pool, data, lookback=back)
    temp = dict()
    temp['model'] = 'vote_top4_linear'
    temp['lookback_days'] = back
    precision, recall, ic, rank_ic, mse = metric_fn(res, score='dyb_score')
    temp['P@3'] = precision[3]
    temp['P@5'] = precision[5]
    temp['P@10'] = precision[10]
    temp['P@30'] = precision[30]
    temp['IC'] = ic
    temp['RankIC'] = rank_ic
    temp['mse'] = mse
    df = df.append(temp, ignore_index=True)

df.to_pickle('ensemble_output/new_find_best_lookback.pkl')
