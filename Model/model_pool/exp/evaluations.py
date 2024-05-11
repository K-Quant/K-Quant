import argparse
import pandas as pd
import numpy as np
from backtest import csi300_industry_map, hot_industry


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


def main(args):
    data = pd.read_pickle(args.predicted_file)
    slc = slice(pd.Timestamp(args.evaluation_start_date), pd.Timestamp(args.evaluation_end_date))
    model_pool_name = data.columns.tolist()
    model_pool_name = [char for char in model_pool_name if char != 'label']
    if args.industry_category != 'all':
        try:
            stock_list = csi300_industry_map[hot_industry[args.industry_category]]
            data = data.loc[(slc, stock_list), :]
            data = data.sort_index(level=0)
        except:
            print("wrong category key, return all stocks")
            data = data[slc]
    else:
        data = data[slc]
    report = []
    for name in model_pool_name:
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
        temp_df = pd.DataFrame(temp, index=[0])
        report.append(temp_df)

    pd_report = pd.concat(report, axis=0)

    pd.to_pickle(pd_report, args.report_file)
    print('finish evaluation')

    return None


def parse_args():
    """
    deliver arguments from json file to program
    :param param_dict: a dict that contains model info
    :return: no return
    """
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--evaluation_start_date', default='2023-04-01')
    parser.add_argument('--evaluation_end_date', default='2024-05-05')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--incremental_mode', default=False, help='load incremental updated models or not')
    parser.add_argument('--industry_category', default='all')
    parser.add_argument('--predicted_file', default='pred_output/da_preds2305.pkl')
    parser.add_argument('--report_file', default='pred_output/da_evaluation.pkl')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    after run this one, all single model, ensemble model result will be store in one pkl file
    """
    args = parse_args()
    main(args)
