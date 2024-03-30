import pandas as pd
import glob
import os
import argparse
from tqdm import tqdm
import re
import datetime
import numpy as np
import sys
import torch


class DataLoader:

    def __init__(self, df, input_len, output_len, batch_size=800, pin_memory=True, device=None):

        column_list = df.columns.tolist()
        input_column = []
        output_column = []
        for i in range(-input_len + 1, 1):
            # pattern = re.compile(rf'(.+?)_{str(i)}')
            input_column += list(filter(re.compile(rf'(.+?)_{str(i)}$').match, column_list))
        for i in range(1, output_len + 1):
            output_column += list(filter(re.compile(rf'(.+?)_{str(i)}$').match, column_list))
        self.df_feature = df[input_column].values
        self.df_label = df[output_column].values
        self.device = device

        if pin_memory:
            self.df_feature = torch.tensor(self.df_feature, dtype=torch.float, device=device)
            self.df_label = torch.tensor(self.df_label, dtype=torch.float, device=device)

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.index = df.index

        self.daily_count = df.groupby(level=0).size().values
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

    def iter_batch(self):
        if self.batch_size <= 0:
            yield from self.iter_daily_shuffle()
            return

        indices = np.arange(len(self.df_label))
        np.random.shuffle(indices)

        for i in range(len(indices))[::self.batch_size]:
            if len(indices) - i < self.batch_size:
                break
            yield i, indices[i:i + self.batch_size]  # NOTE: advanced indexing will cause copy

    def iter_daily_shuffle(self):
        indices = np.arange(len(self.daily_count))
        np.random.shuffle(indices)
        for i in indices:
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

    def get(self, slc):
        # now get only output two items
        outs = self.df_feature[slc], self.df_label[slc]
        # outs = self.df_feature[slc], self.df_label[slc]
        mask = self.padding_mask(outs[0])

        if not self.pin_memory:
            outs = tuple(torch.tensor(x, device=self.device) for x in outs)

        return outs + (self.index[slc], mask,)

    def padding_mask(self, features, max_len=None):
        # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
        lengths = [X.shape[0] for X in features]  # original sequence length for each time series
        if max_len is None:
            max_len = max(lengths)

        padding_masks = self._padding_mask(torch.tensor(lengths, dtype=torch.int16, device=self.device), max_len=max_len)
        # (batch_size, padded_length) boolean tensor, "1" means keep
        return padding_masks

    @staticmethod
    def _padding_mask(lengths, max_len=None):
        """
        Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
        where 1 means keep element at this position (time step)
        """
        batch_size = lengths.numel()
        max_len = max_len or lengths.max_val()
        return (torch.arange(0, max_len, device=lengths.device)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))


def one_symbol_loader(args, file):
    x = pd.read_csv(file).set_index('date')
    for column in ['open', 'close', 'high', 'low', 'volume', 'amount', '振幅', '涨跌幅', '涨跌额', '换手率']:
        x[column] = (x[column] - x[column].mean()) / (x[column].std()+0.001)
    shift_dict = [x]
    index_file = x.index.tolist()[args.input_seq_len - 1: -args.output_seq_len]
    for index in range(args.input_seq_len + args.output_seq_len - 1):
        temp = x.shift(-1)
        x = temp
        shift_dict.append(temp)

    symbol = re.findall(rf'{args.data_folder}/(.+?).csv', file)[0]
    index_file = [datetime.datetime.strptime(i, '%Y-%m-%d') for i in index_file]
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
    # normalize volume and amount to compress disk space they needed
    # seperate normalize input and output
    return result


def create_ltsf_loader(args, device):
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
    assert args.train_start_date < args.train_end_date
    assert args.valid_start_date < args.valid_end_date
    assert args.test_start_date < args.test_end_date
    file_dict = glob.glob(os.path.join(args.data_folder, "*.csv"))
    one_symbol_set = []
    print('building data...')
    for file in tqdm(file_dict):
        # index file need to used to tell dataloaders when to change source file
        temp = one_symbol_loader(args, file).astype('float32')
        # temp = temp.loc[lambda x: (x.index.get_level_values('date') < quarter_time_str) &
        # (x.index.get_level_values('date') >= args.train_start_date)]
        one_symbol_set.append(temp)
    result = pd.concat(one_symbol_set, axis=0)
    print('sorting by time stamp...')
    result = result.sort_index(level='date')
    slc = slice(pd.Timestamp(args.train_start_date), pd.Timestamp(args.train_end_date))
    df_train = result[slc]
    train_loader = DataLoader(df_train, args.input_seq_len, args.output_seq_len, batch_size=args.batch_size, device=device)
    slc = slice(pd.Timestamp(args.valid_start_date), pd.Timestamp(args.valid_end_date))
    df_valid = result[slc]
    valid_loader = DataLoader(df_valid, args.input_seq_len, args.output_seq_len, batch_size=args.batch_size, device=device)
    slc = slice(pd.Timestamp(args.test_start_date), pd.Timestamp(args.test_end_date))
    df_test = result[slc]
    test_loader = DataLoader(df_test, args.input_seq_len, args.output_seq_len, batch_size=args.batch_size, device=device)

    return train_loader, valid_loader, test_loader


def test_fake_epoch(train_loader):
    for i, slc in tqdm(train_loader.iter_batch(), total=train_loader.batch_length):
        feature, label, stock_index, mask = train_loader.get(slc)
        print(feature.shape)
        print(label.shape)
        print(mask.shape)
        print(stock_index.shape)
        break
    return None


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
        parser.add_argument('--input_seq_len', default=60)
        parser.add_argument('--output_seq_len', default=10)
        parser.add_argument('--device', default='cpu')
        args = parser.parse_args()
        return args


    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    train_loader, valid_loader, test_loader = create_ltsf_loader(args, device=device)
    test_fake_epoch(train_loader)

