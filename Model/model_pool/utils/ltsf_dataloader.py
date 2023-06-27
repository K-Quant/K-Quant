import pandas as pd
import glob
import os
import argparse
from tqdm import tqdm
import re
import datetime
import numpy as np
import sys


def one_symbol_loader(args, file):
    x = pd.read_csv(file).set_index('date')
    shift_dict = [x]
    index_file = x.index.tolist()[args.input_seq_len - 1: -args.output_seq_len]
    for index in range(args.input_seq_len + args.output_seq_len - 1):
        temp = x.shift(-1)
        x = temp
        shift_dict.append(temp)

    symbol = re.findall(rf'{args.data_folder}/(.+?).csv', file)[0]
    multi_index_array = [index_file,
                         [symbol for i in index_file]]
    multi_index_tuple = list(zip(*multi_index_array))
    index = pd.MultiIndex.from_tuples(multi_index_tuple, names=['date', 'symbol'])
    result = pd.concat(shift_dict, axis=1, ignore_index=True).dropna()
    result.index = index
    column_name = []
    for i in range(-args.input_seq_len + 1, args.output_seq_len + 1):
        for column in ['open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude',
                       'change_ratio', 'change', 'turnover']:
            column_name.append(f"{column}_{i}")
    result.columns = column_name
    return result


def one_symbol_tuple(args, file):
    """
    :param args:
    :param file: the csv file name
    :return: a numpy array that contains time stamp and symbol shape is (N, (args.in_len+args.out_len)*d_feat+2)
    the "2" is the time stamp and symbol for every ts
    N = day_num-args.in_len-args._out_len+1
    """
    x = pd.read_csv(file).set_index('date')
    shift_dict = [x]
    index_file = x.index.tolist()[args.input_seq_len - 1: -args.output_seq_len]
    for index in range(args.input_seq_len + args.output_seq_len - 1):
        temp = x.shift(-1)
        x = temp
        shift_dict.append(temp)

    symbol = re.findall(rf'{args.data_folder}/(.+?).csv', file)[0]
    multi_index_array = [index_file,
                         [symbol for i in index_file]]
    multi_index_array = np.array(multi_index_array).transpose()
    result = pd.concat(shift_dict, axis=1, ignore_index=True).dropna()
    result = np.concatenate((multi_index_array, np.array(result)), axis=1)
    return result


def create_ltsf_loader(args):
    """
    in create ltsf loader, we need to build a dataset that suits for long-term time series forecasting
    :param args:
        args.train_start_date
        args.train_end_date
        args.valid_start_date
        args.valid_end_date
        args.test_start_date
        args.test_end_date
        args.input_seq_len
        args.output_seq_len
        args.data_folder
        args.stock_dict
    :return:
        dataloaders, that every batch contains the time series of different stocks in the same day and the
    pred is also time series
    """
    train_start_time = datetime.datetime.strptime(args.train_start_date, '%Y-%m-%d')
    train_end_time = datetime.datetime.strptime(args.train_end_date, '%Y-%m-%d')
    valid_start_time = datetime.datetime.strptime(args.valid_start_date, '%Y-%m-%d')
    valid_end_time = datetime.datetime.strptime(args.valid_end_date, '%Y-%m-%d')
    test_start_time = datetime.datetime.strptime(args.test_start_date, '%Y-%m-%d')
    test_end_time = datetime.datetime.strptime(args.test_end_date, '%Y-%m-%d')
    assert train_start_time < train_end_time
    assert valid_start_time < valid_end_time
    assert test_start_time < test_end_time
    middle_time = train_start_time+(train_end_time-train_start_time)//2
    quarter_time = middle_time + (train_end_time - middle_time)//2
    quarters_time = train_start_time + (middle_time - train_start_time)//2
    middle_time_str = middle_time.strftime('%Y-%m-%d')
    quarter_time_str = quarter_time.strftime('%Y-%m-%d')
    quarters_time_str = quarters_time.strftime('%Y-%m-%d')
    print(f'{quarter_time}, {middle_time}, {quarters_time}')
    file_dict = glob.glob(os.path.join(args.data_folder, "*.csv"))
    one_symbol_set = []
    print('building dataframe...training chunk 1')
    for file in file_dict:
        temp = one_symbol_loader(args, file)
        print(temp.index.get_level_values('date') > train_start_time)
        print(temp.loc((temp.index < quarter_time) & (temp.index >= train_start_time)))
        one_symbol_set.append(temp)
        break
    result = pd.concat(one_symbol_set, axis=0)
    result = result.sort_index(level='date')
    return result


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_start_date', default='2007-01-01')
        parser.add_argument('--train_end_date', default='2014-12-31')
        parser.add_argument('--valid_start_date', default='2015-01-01')
        parser.add_argument('--valid_end_date', default='2016-12-31')
        parser.add_argument('--test_start_date', default='2017-01-01')
        parser.add_argument('--test_end_date', default='2020-12-31')
        parser.add_argument('--data_folder', default='../../csv_data')
        parser.add_argument('--input_seq_len', default=2)
        parser.add_argument('--output_seq_len', default=2)
        args = parser.parse_args()
        return args

    args = parse_args()
    create_ltsf_loader(args)