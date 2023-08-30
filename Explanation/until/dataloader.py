import pickle
import torch
import numpy as np
import pandas as pd

# import qlib
# import datetime
# from qlib.config import REG_CN
#
# provider_uri = './qlib_bin'
# qlib.init(provider_uri=provider_uri, region=REG_CN)
# from qlib.data.dataset import DatasetH
# from qlib.data.dataset.handler import DataHandlerLP

class DataLoader:

    def __init__(self, df_feature, df_label, df_market_value, df_stock_index, batch_size=800, pin_memory=True,
                 start_index=0, device=None):

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
        self.pin_memory = pin_memory
        self.start_index = start_index
        a = df_label.groupby(level=0)

        self.daily_count = df_label.groupby(level=0).size().values
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
        outs = self.df_feature[slc], self.df_label[slc][:, 0], self.df_market_value[slc], self.df_stock_index[slc]
        # outs = self.df_feature[slc], self.df_label[slc]

        if not self.pin_memory:
            outs = tuple(torch.tensor(x, device=self.device) for x in outs)

        return outs + (self.index[slc],)


def load_data_df(args):
    data_path = r'{}/data/alpha360.pkl'.format(args.data_root)
    with open(data_path, "rb") as fh:
        df_total = pickle.load(fh)
    slc = slice(pd.Timestamp(args.start_date), pd.Timestamp(args.end_date))
    data_df = df_total[slc]
    return data_df


def create_data_loaders(args):
    """
    load qlib alpha360 data and split into train, validation and test loader
    :param args:
    :return:
    """
    # split those three dataset into train, valid and test
    with open(args.market_value_path, "rb") as fh:
        # load market value
        df_market_value = pickle.load(fh)

    df_check = load_data_df(args)
    df_market_value = df_market_value / 1000000000
    # market value of every day from 07 to 20
    stock_index = np.load(args.stock_index, allow_pickle=True).item()
    start_index = 0
    slc = slice(pd.Timestamp(args.start_date), pd.Timestamp(args.end_date))
    df_check['market_value'] = df_market_value[slc]
    df_check['market_value'] = df_check['market_value'].fillna(df_check['market_value'].mean())
    df_check['stock_index'] = 733
    df_check['stock_index'] = df_check.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    # the market value and stock_index added to each line
    data_loader = DataLoader(df_check["feature"], df_check["label"], df_check['market_value'], df_check['stock_index'],
                             batch_size=-1, pin_memory=True, start_index=start_index,
                             device='cpu')
    return data_loader


def get_daily_inter(df, shuffle=False, step_size=1):
    # organize the train data into daily batches
    daily_count = df.groupby(level=0).size().values
    daily_index = np.roll(np.cumsum(daily_count), 1)
    daily_index[0] = 0

    stocks_daily_ = []
    for date, stocks in pd.Series(index=df.index, dtype=np.float32).groupby("datetime"):
        stocks_daily_.append(list(stocks.loc[date, :].index))

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

    return daily_index, daily_count, stocks_daily


# def create_data_loaders(args):
#     """
#     return a single dataloader for prediction
#     """
#     start_time = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
#     end_time = datetime.datetime.strptime(args.end_date, '%Y-%m-%d')
#     start_date = args.start_date
#     end_date = args.end_date
#     # 此处fit_start_time参照官方文档和代码
#     hanlder = {'class': 'Alpha360', 'module_path': 'qlib.contrib.data.handler',
#                'kwargs': {'start_time': start_time, 'end_time': end_time, 'fit_start_time': start_time,
#                           'fit_end_time': end_time, 'instruments': 'csi300','infer_processors': [
#                        {'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True}},
#                        {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}],
#                           'learn_processors': [{'class': 'DropnaLabel'},
#                                                {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}}],
#                           'label': ['Ref($close, -2) / Ref($close, -1) - 1']}}
#     segments = {'total': (start_date, end_date)}
#     dataset = DatasetH(hanlder, segments)
#     # prepare return a list of df, df_test is the first one
#     df_total = dataset.prepare(["total"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L, )[0]
#     # ----------------------------------------
#     import pickle5 as pickle
#     # only HIST need this
#     with open(args.market_value_path, "rb") as fh:
#         # load market value
#         df_market_value = pickle.load(fh)
#         # the df_market_value save
#     df_market_value = df_market_value / 1000000000
#     stock_index = np.load(args.stock_index, allow_pickle=True).item()
#     # stock_index is a dict and stock is the key, index is the value
#     start_index = 0
#
#     slc = slice(pd.Timestamp(start_date), pd.Timestamp(end_date))
#     df_check = df_total[slc]
#     df_check['market_value'] = df_market_value[slc]
#     df_check['market_value'] = df_check['market_value'].fillna(df_check['market_value'].mean())
#     df_check['stock_index'] = 733
#     df_check['stock_index'] = df_check.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
#
#     data_loader = DataLoader(df_check["feature"], df_check["label"], df_check['market_value'], df_check['stock_index'],
#                              batch_size=-1, pin_memory=True, start_index=start_index, device='cpu')
#     return data_loader , df_check
