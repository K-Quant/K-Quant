# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import random
import warnings

import qlib
from qlib.data.dataset import DataHandlerLP

warnings.filterwarnings("ignore")
import sys
from pathlib import Path

import copy
import os.path
from tqdm import tqdm
import time
import sys

# sys.path.insert(0, sys.path[0]+"/../")
DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.insert(0, str(DIRNAME.parent))
sys.path.insert(0, str(DIRNAME.parent.parent))
from model_pool.models.model import MLP, GRU, LSTM, GAT, ALSTM, SFM, RSR, HIST, KEnhance
from pprint import pprint
from typing import Optional, Dict, Union, List
import argparse
import pandas as pd
import torch
import yaml
import json
import numpy as np

from model_pool.exp.learn import relation_model_dict

from models.DoubleAdapt_model.model import IncrementalManager, DoubleAdaptManager
from models.DoubleAdapt_model import utils

from learn import get_model

# def get_model(model_name):
#     if model_name.upper() == 'MLP':
#         return MLP
#     if model_name.upper() == 'LSTM':
#         return LSTM
#     if model_name.upper() == 'GRU':
#         return GRU
#     if model_name.upper() == 'GATS':
#         return GAT
#     if model_name.upper() == 'SFM':
#         return SFM
#     if model_name.upper() == 'ALSTM':
#         return ALSTM
#     if model_name.upper() == 'RSR':
#         return RSR
#     if model_name.upper() == 'HIST':
#         return HIST
#     if model_name.upper() == 'KEnhance':
#         return KEnhance
#     raise ValueError('unknown model name `%s`' % model_name)


class IncrementalExp:
    """
    Example:
        .. code-block:: python

            python -u main.py run_all --forecast_model GRU --market csi300 --data_dir crowd_data --rank_label False \
            --first_order True --adapt_x True --adapt_y True --num_head 8 --tau 10 \
            --lr 0.001 --lr_da 0.01 --online_lr "{'lr': 0.001, 'lr_da': 0.001, 'lr_ma': 0.001}"
    """

    def __init__(
            self,
            args,
            data_dir="cn_data",
            calendar_path=None,
            market="csi300",
            horizon=1,
            alpha=360,
            x_dim=None,
            step=20,
            lr=0.001,
            lr_ma=None,
            lr_da=0.01,
            lr_x=None,
            lr_y=None,
            online_lr: dict = None,
            early_stop: int = 10,
            reg=0.5,
            weight_decay=0,
            num_head=8,
            tau=10,
            first_order=True,
            adapt_x=True,
            adapt_y=True,
            naive=False,
            preprocess_tensor=True,
            use_extra=False,
            tag=None,
            rank_label=False,
            h_path=None,
            skip_valid_epoch=5,
            relation_path=None,
            stock_index_path=None,
    ):
        """
        Args:
            data_dir (str):
                source data dictionary under root_path
            calendar_path (str):
                the path of calendar. If None, use '~/.qlib/qlib_data/cn_data/calendar/days.txt'.
            market (str):
                'csi300' or 'csi500'
            horizon (int):
                define the stock price trend
            alpha (int):
                360 or 158
            x_dim (int):
                the dimension of stock features (e.g., factor_num * time_series_length)
            step (int):
                incremental task interval, i.e., timespan of incremental data or test data
            lr (float):
                learning rate of forecast model
            lr_ma (float):
                learning rate of model adapter. If None, use lr.
            lr_da (float):
                learning rate of data adapter
            lr_x (float):
                if both lr_x and lr_y are not None, specify the learning rate of the feature adaptation layer.
            lr_y (float):
                if both lr_x and lr_y are not None, specify the learning rate of the label adaptation layer.
            online_lr (dict):
                learning rates during meta-valid and meta-test. Example: --online lr "{'lr_da': 0, 'lr': 0.0001}".
            reg (float):
                regularization strength
            weight_decay (float):
                L2 regularization of the (Adam) optimizer
            num_head (int):
                number of transformation heads
            tau (float):
                softmax temperature
            first_order (bool):
                whether use first-order approximation version of MAML
            adapt_x (bool):
                whether adapt features
            adapt_y (bool):
                whether adapt labels
            naive (bool):
                if True, degrade to naive incremental baseline; if False, use DoubleAdapt
            preprocess_tensor (bool):
                if False, temporally transform each batch from `numpy.ndarray` to `torch.Tensor` (slow, not recommended)
            use_extra (bool):
                if True, use extra segments for upper-level optimization (not recommended when step is large enough)
            tag (str):
                to distinguish experiment id
            h_path (str):
                prefetched handler file path to load
        """
        self.data_dir = data_dir
        self.provider_uri = os.path.join(args.root_path, data_dir)
        # qlib.init(provider_uri=self.provider_uri, region='cn')

        if calendar_path is None:
            calendar_path = os.path.join(args.root_path, data_dir, 'calendars/day.txt')
        calendar = pd.read_csv(calendar_path, header=None)[0]  # the column name is 0 for .txt files
        self.ta = utils.TimeAdjuster(calendar)

        self.market = market
        if self.market == "csi500":
            self.benchmark = "SH000905"
        else:
            self.benchmark = "SH000300"
        self.step = step
        self.horizon = horizon
        self.model_name = args.model_name  # downstream forecasting models' type
        self.alpha = alpha
        self.tag = tag
        if self.tag is None:
            self.tag = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
        self.rank_label = rank_label
        self.lr = lr
        self.lr_da = lr_da
        self.lr_ma = lr if lr_ma is None else lr_ma
        self.lr_x = lr_x
        self.lr_y = lr_y
        if online_lr is not None and 'lr' in online_lr:
            online_lr['lr_model'] = online_lr['lr']
        self.online_lr = online_lr
        self.early_stop = early_stop
        self.num_head = num_head
        self.temperature = tau
        self.first_order = first_order
        self.naive = naive
        self.adapt_x = adapt_x
        self.adapt_y = adapt_y
        self.reg = reg
        self.weight_decay = weight_decay
        self.not_sequence = self.model_name in ["MLP", 'Linear'] and self.alpha == 158
        self.skip_valid_epoch = skip_valid_epoch

        # FIXME: Override the segments!!
        # self.segments = {
        #         'train': ('2008-01-01', '2014-12-31'),
        #         'valid': ('2015-01-01', '2016-12-31'),
        #         'test': ('2017-01-01', '2020-08-01')
        # }
        self.segments = {
            'train': (args.incre_train_start, args.incre_train_end),
            'valid': (args.incre_val_start, args.incre_val_end),
            'test': (args.test_start, args.test_end)
        }
        print(self.segments)
        for k, v in self.segments.items():
            self.segments[k] = (self.ta.align_time(self.segments[k][0], tp_type='start'),
                                self.ta.align_time(self.segments[k][1], tp_type='end'))

        # self.test_slice = slice(self.ta.align_time(test_start, tp_type='start'), self.ta.align_time(test_end, tp_type='end'))
        self.test_slice = slice(self.ta.align_time(self.segments['test'][0], tp_type='start'),
                                self.ta.align_time(self.segments['test'][1], tp_type='end'))
        self.early_stop = args.early_stop
        self.h_path = h_path
        self.preprocess_tensor = preprocess_tensor
        self.use_extra = use_extra

        self.factor_num = 6 if self.alpha == 360 else 20
        self.x_dim = x_dim if x_dim else (360 if self.alpha == 360 else 20 * 20)
        # print('Experiment name:', self.experiment_name)

        self.relation_matrix = torch.tensor(np.load(relation_path), dtype=torch.float32) if relation_path else None
        self.stock_index_table = np.load(stock_index_path, allow_pickle=True).item() if stock_index_path else None
        self.day_by_day = False

    @property
    def experiment_name(self):
        return f"{self.market}_{self.model_name}_alpha{self.alpha}_horizon{self.horizon}_step{self.step}" \
               f"_rank{self.rank_label}_{self.tag}"

    def _init_model(self, args):
        param_dict = json.load(open(args.model_path + args.model_name + '/info.json'))['config']
        param_dict['model_dir'] = args.model_path

        print('load model ', param_dict['model_name'])
        if param_dict['model_name'] == 'SFM':
            model = get_model(param_dict['model_name'])(d_feat=param_dict['d_feat'], output_dim=32, freq_dim=25,
                                                        hidden_size=param_dict['hidden_size'],
                                                        dropout_W=0.5, dropout_U=0.5, device=args.device)
        elif param_dict['model_name'] == 'ALSTM':
            model = get_model(param_dict['model_name'])(param_dict['d_feat'], param_dict['hidden_size'],
                                                        param_dict['num_layers'], param_dict['dropout'], 'LSTM')
        elif param_dict['model_name'] == 'Transformer':
            model = get_model(param_dict['model_name'])(param_dict['d_feat'], param_dict['hidden_size'],
                                                        param_dict['num_layers'], dropout=0.5)
        elif param_dict['model_name'] == 'HIST':
            model = get_model(param_dict['model_name'])(d_feat=param_dict['d_feat'], num_layers=param_dict['num_layers']
                                                        , K=param_dict['K'])
            self.day_by_day = True
        elif param_dict['model_name'] in relation_model_dict:
            num_relation = self.relation_matrix.shape[2]  # the number of relations
            model = get_model(param_dict['model_name'])(num_relation=num_relation, d_feat=param_dict['d_feat'],
                                                        num_layers=param_dict['num_layers'])
            self.day_by_day = True
        else:
            model = get_model(param_dict['model_name'])(d_feat=param_dict['d_feat'],
                                                        num_layers=param_dict['num_layers'])

        if isinstance(model, GAT):
            self.day_by_day = True

        model.to(args.device)
        # model.load_state_dict(torch.load(param_dict['model_dir'] + '/model.bin', map_location=args.device))
        # model = get_model('MLP')(d_feat=param_dict['d_feat'], num_layers=param_dict['num_layers'])
        # model.load_state_dict(torch.load(param_dict['model_dir'] + '/model.bin', map_location=args.device))
        return model

    def offline_training(self, args, segments: Dict[str, tuple] = None, data: pd.DataFrame = None,
                         reload: bool = False, model_save_path=None):
        model = self._init_model(args)
        model.to(args.device)

        if self.naive:
            framework = IncrementalManager(model, x_dim=self.x_dim, lr_model=self.lr,
                                           begin_valid_epoch=0, over_patience=self.early_stop,
                                           day_by_day=self.day_by_day,
                                           stock_index_table=self.stock_index_table,
                                           relation_matrix=self.relation_matrix)
        else:
            print("DoubleAdaptManager")
            framework = DoubleAdaptManager(model, x_dim=self.x_dim, lr_model=self.lr, weight_decay=self.weight_decay,
                                           first_order=self.first_order, begin_valid_epoch=self.skip_valid_epoch,
                                           factor_num=self.factor_num,
                                           lr_da=self.lr_da, lr_ma=self.lr_ma, online_lr=self.online_lr,
                                           lr_x=self.lr_x, lr_y=self.lr_y, over_patience=self.early_stop,
                                           adapt_x=self.adapt_x, adapt_y=self.adapt_y, reg=self.reg,
                                           num_head=self.num_head, temperature=self.temperature,
                                           day_by_day=self.day_by_day,
                                           stock_index_table=self.stock_index_table,
                                           relation_matrix=self.relation_matrix)
        if reload and os.path.exists(model_save_path):
            framework.load_state_dict(torch.load(model_save_path))
            print('Reload checkpoint from', model_save_path)
        else:
            if args.resume and os.path.exists(model_save_path):
                framework.load_state_dict(torch.load(model_save_path))
                print('Reload checkpoint from', model_save_path)
            if segments is None:
                segments = self.segments
            # rolling_tasks = utils.organize_all_tasks(segments,
            #                                          self.ta, step=self.step, trunc_days=self.horizon + 1,
            #                                          rtype=utils.TimeAdjuster.SHIFT_SD, use_extra=self.use_extra)
            all_rolling_tasks = utils.organize_tasks(segments['train'][0], segments['valid'][-1], self.ta, self.step,
                                                     trunc_days=self.horizon + 1,
                                                     rtype=utils.TimeAdjuster.SHIFT_SD, use_extra=self.use_extra)
            rolling_tasks = {}
            rolling_tasks['train'], rolling_tasks['valid'] = utils.split_rolling_tasks(all_rolling_tasks,
                                                                                       split_point=segments['train'][
                                                                                           -1])
            rolling_tasks_data = {k: utils.get_rolling_data(rolling_tasks[k],
                                                            # data=self._load_data() if data is None else data,
                                                            data=data,
                                                            factor_num=self.factor_num, horizon=self.horizon,
                                                            not_sequence=self.not_sequence,
                                                            sequence_last_dim=self.alpha == 360,
                                                            to_tensor=self.preprocess_tensor)
                                  for k in ['train', 'valid']}
            framework.fit(meta_tasks_train=rolling_tasks_data['train'],
                          meta_tasks_val=rolling_tasks_data['valid'],
                          checkpoint_path=model_save_path)
        return framework

    def online_training(self, args, segments: Dict[str, tuple] = None,
                        data: pd.DataFrame = None, framework=None, ):
        """
        Perform incremental learning on the test data.

        Args:
            segments (Dict[str, tuple]):
                The date range of training data, validation data, and test data.
                Example::
                    {
                        'train': ('2008-01-01', '2014-12-31'),
                        'valid': ('2015-01-01', '2016-12-31'),
                        'test': ('2017-01-01', '2020-08-01')
                    }
            data (pd.DataFrame):
                the index col is pd.MultiIndex with the datetime as level 0 and the stock ID as level 1;
                the col named 'feature' contains the stock feature vectors;
                the col named 'label' contains the ground-truth labels.
            model_save_path (str):
                if not None, reload checkpoints

        Returns:
            pd.DataFrame:
                the index col is pd.MultiIndex with the datetime as level 0 and the stock ID as level 1;
                the col named 'pred' contains the predictions of the model;
                the col named 'label' contains the ground-truth labels which have been preprocessed and may not be the raw.
        """
        if framework is None:
            model = self._init_model(args)
            model.to(args.device)
            if self.naive:
                framework = IncrementalManager(model, x_dim=self.x_dim, lr_model=self.lr,
                                               online_lr=self.online_lr, weight_decay=self.weight_decay,
                                               first_order=True, alpha=self.alpha, begin_valid_epoch=0,
                                               day_by_day=self.day_by_day,
                                               stock_index_table=self.stock_index_table,
                                               relation_matrix=self.relation_matrix)
            else:
                framework = DoubleAdaptManager(model, x_dim=self.x_dim, lr_model=self.lr,
                                               weight_decay=self.weight_decay,
                                               first_order=self.first_order, begin_valid_epoch=0,
                                               factor_num=self.factor_num,
                                               lr_da=self.lr_da, lr_ma=self.lr_ma, online_lr=self.online_lr,
                                               lr_x=self.lr_x, lr_y=self.lr_y,
                                               adapt_x=self.adapt_x, adapt_y=self.adapt_y, reg=self.reg,
                                               num_head=self.num_head, temperature=self.temperature,
                                               day_by_day=self.day_by_day,
                                               stock_index_table=self.stock_index_table,
                                               relation_matrix=self.relation_matrix)
            if args.reload and args.model_save_path is not None:
                framework.load_state_dict(torch.load(args.model_save_path))
                print('Reload checkpoint from', args.model_save_path)
        else:
            model = self._init_model(args)
            model.to(args.device)

        if segments is None:
            segments = self.segments
        # rolling_tasks = utils.organize_all_tasks(segments, self.ta, step=self.step, trunc_days=self.horizon + 1,
        #                                          rtype=utils.TimeAdjuster.SHIFT_SD, use_extra=self.use_extra)
        rolling_tasks = utils.organize_tasks(segments['valid'][0], segments['test'][-1], self.ta, self.step,
                                             trunc_days=self.horizon + 1,
                                             rtype=utils.TimeAdjuster.SHIFT_SD, use_extra=self.use_extra)
        rolling_tasks_data = utils.get_rolling_data(rolling_tasks,
                                                    # data=self._load_data() if data is None else data,
                                                    data=data,
                                                    factor_num=self.factor_num, horizon=self.horizon,
                                                    not_sequence=self.not_sequence,
                                                    sequence_last_dim=self.alpha == 360,
                                                    to_tensor=self.preprocess_tensor)

        return framework.inference(meta_tasks_test=rolling_tasks_data, date_slice=self.test_slice)

    def _evaluate_metrics(self, pred: pd.DataFrame, name, label_all=None):
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP

        """        
        Note that the labels in pred_y_all are preprocessed. IC should be calculated by the raw labels. 
        """

        if label_all is None:
            ds = DatasetH({'class': 'Alpha360', 'module_path': 'qlib.contrib.data.handler',
                           'kwargs': {'fit_start_time': self.segments['train'][0],
                                      'fit_end_time': self.segments['train'][1], 'instruments': 'csi300'
                                      }},
                          segments={'test': self.segments['test']})
            label_all = ds.prepare(segments="test", col_set="label", data_key=DataHandlerLP.DK_R)
            label_all = label_all.dropna(axis=0)
        # print("pred", pred)
        # print("label_all", label_all)
        df = pred.loc[label_all.index]
        df['label'] = label_all.values

        ic = df.groupby('datetime').apply(lambda df: df["pred"].corr(df["label"]))
        ric = df.groupby('datetime').apply(lambda df: df["pred"].corr(df["label"], method="spearman"))
        metrics = {
            "IC": ic.mean(),
            "ICIR": ic.mean() / ic.std(),
            "Rank IC": ric.mean(),
            "Rank ICIR": ric.mean() / ric.std(),
        }
        print(name)
        pprint(metrics)
        return df

    def workflow(self, args):
        from model_pool.utils.dataloader import create_doubleadapt_loaders
        ds = create_doubleadapt_loaders(args, self.rank_label)
        data = ds.prepare(['train'], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L, )[0]

        print(self.segments)
        assert data.index[0][0] <= self.ta.align_time(self.segments['train'][0], tp_type='start'), print(data.index[0][0])
        # assert data.index[-1][0] >= self.ta.align_time(self.segments['test'][-1], tp_type='end'), print(data.index[-1][0], self.ta.align_time(self.segments['test'][-1], tp_type='end'))
        print("offline_training")
        framework = self.offline_training(args=args, data=data, reload=args.reload, model_save_path=args.model_save_path)

        print("evaluation")

        if not args.no_test:
            # args.online_lr['lr_model'] = 0.0002
            # framework.lr_model = 0.0002
            # args.online_lr['lr_ma'] = 0.0001
            # framework.framework.opt.param_groups[0]['lr'] = args.online_lr['lr_ma']
            # data = ds.prepare(['train'], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L, )[0]
            pred_y_all_incre = self.online_training(data=data, framework=framework, args=args)

            label_all = ds.prepare(segments="train", col_set="label", data_key=DataHandlerLP.DK_R)
            label_all = label_all.loc(axis=0)[self.test_slice].dropna(axis=0)
            pred_y_all_incre = self._evaluate_metrics(pred_y_all_incre, "incre_model", label_all)
            # pred_y_all_basic = self._evaluate_metrics(pred_y_all_basic, "basic_model", label_all)
            if not os.path.exists(args.result_path):
                os.makedirs(args.result_path)


            year = int(args.test_start[:4])
            Q = (int(args.test_start[5:7]) - 1) // 3 + 1
            pred_y_all_incre.to_csv(os.path.join(args.result_path, f"DoubleAdapt_{args.model_name}_{year}Q{Q}.csv"))


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model_name', default='GRU')
    parser.add_argument('--rank_label', type=str_to_bool, default=True)
    parser.add_argument('--naive', type=str_to_bool, default=False)
    parser.add_argument('--adapt_y', type=str_to_bool, default=True)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_ma', type=float, default=None)
    parser.add_argument('--lr_da', type=float, default=0.01)
    parser.add_argument('--online_lr', type=str, default=None)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--skip_valid_epoch', type=int, default=1)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--result_path', default="./pred_output/")
    parser.add_argument('--model_path', default='./output/', help='learned model')
    parser.add_argument('--model_save_path', default="./output/INCRE/", help='updated model')
    parser.add_argument('--root_path', default='~/.qlib/qlib_data', help='')
    parser.add_argument('--incre_train_start', default='2008-01-01')
    parser.add_argument('--incre_train_end', default='2019-12-31')
    parser.add_argument('--incre_val_start', default='2020-01-01')
    parser.add_argument('--incre_val_end', default='2022-12-31')
    parser.add_argument('--test_start', default='2021-01-01')
    parser.add_argument('--test_end', default='2023-06-30')
    parser.add_argument('--reload', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--no_test', action='store_true', default=False)

    # input for csi 300
    parser.add_argument('--stock2concept_matrix', default='../data/csi300_stock2concept.npy')
    parser.add_argument('--stock2stock_matrix', default='../data/csi300_multi_stock2stock_hidy_2023.npy')
    parser.add_argument('--stock_index', default='../data/csi300_stock_index.npy')

    args = parser.parse_args()

    if args.model_name in ['GRU', 'LSTM', 'ALSTM', 'SFM', 'MLP', 'GATs']:
        args.stock2concept_matrix = None
        args.stock2stock_matrix = None

    if args.no_test:
        args.test_end = args.incre_val_end
        args.test_start = args.test_end

    if args.online_lr is not None:
        args.online_lr = eval(args.online_lr)

    retrain_segs = [('01-01', '03-31'), ('04-01', '06-30'), ('07-01', '09-30'), ('10-01', '12-31')]
    if args.reload:
        # Require args for test. Ignore args for validation.
        args.Q = ((int(args.test_end[5:7]) - 1) // 3 - 1) % 4 + 1
        args.year = int(args.test_end[:4]) - (args.Q == 4)
        args.incre_val_start = f'{args.year}-{retrain_segs[args.Q - 1][0]}'
        args.incre_val_end = f'{args.year}-{retrain_segs[args.Q - 1][1]}'
        args.test_start = f'{args.year}-{retrain_segs[args.Q][0]}'
    else:
        # Require arguments for validation.
        args.Q = (int(args.incre_val_start[5:7]) - 1) // 3 + 1
        args.year = int(args.incre_val_start[:4])
        # assert args.incre_val_start[5:] == retrain_segs[args.Q - 1][0], print(args.incre_val_start[5:], retrain_segs[args.Q - 1][0])
        # assert args.incre_val_end[5:] == retrain_segs[args.Q - 1][1], print(args.incre_val_end[5:], retrain_segs[args.Q - 1][1])

    online_lr_str = ''
    if args.online_lr is not None:
        for k, v in args.online_lr.items():
            online_lr_str += f'_online_{k}_{v}'
    # checkpoint_name = f'{args.model_name}_DoubleAdapt_step{args.step}_{args.year}Q{args.Q}_lr{args.lr}_ma{args.lr_ma}_da{args.lr_da}{online_lr_str}.bin'
    # checkpoint_name = f'{args.model_name}_DoubleAdapt_step{args.step}_{args.year}Q{args.Q}_lr{args.lr}_ma{args.lr_ma}_da{args.lr_da}_online_lr_da_{args.lr_da / 10}.bin'
    checkpoint_name = f'{args.model_name}_DoubleAdapt_step{args.step}_{args.year}Q{args.Q}_lr{args.lr}_ma{args.lr_ma}_da{args.lr_da}.bin'
    if args.model_path is not None:
        args.model_save_path = os.path.join(args.model_path, args.model_name + '_DoubleAdapt')
        if not os.path.exists(args.model_save_path):
            os.makedirs(args.model_save_path)
        args.model_save_path = os.path.join(args.model_save_path, checkpoint_name)

    if args.reload and not os.path.exists(args.model_save_path):
        print(f"Need retraining! No checkpoint:", args.model_save_path)

    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.benchmark=False
    # torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    args = parse_args()
    print(args)
    setup_seed(1)
    a = IncrementalExp(args=args, data_dir='cn_data', rank_label=args.rank_label, adapt_y=args.adapt_y,
                       naive=args.naive, early_stop=args.early_stop, step=args.step,
                       skip_valid_epoch=args.skip_valid_epoch,
                       lr=args.lr, lr_ma=args.lr_ma, lr_da=args.lr_da, online_lr=args.online_lr,
                       relation_path=args.stock2concept_matrix if args.model_name == 'HIST' else args.stock2stock_matrix,
                       stock_index_path=args.stock_index)
    a.workflow(args=args)
