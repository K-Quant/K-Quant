# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import warnings
warnings.filterwarnings("ignore")
import copy
import os.path
from tqdm import tqdm
import time
import sys
sys.path.insert(0, sys.path[0]+"/../")
from models.model import MLP, GRU, LSTM, GAT, ALSTM, SFM
from utils.dataloader import create_doubleadapt_loaders
from pathlib import Path
from pprint import pprint
from typing import Optional, Dict, Union, List
import argparse
import sys
import pandas as pd
import torch
import yaml
import json
import numpy as np
DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME))
sys.path.append(str(DIRNAME.parent.parent.parent))
from models.DoubleAdapt_model.model import IncrementalManager, DoubleAdaptManager
from models.DoubleAdapt_model import utils


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
    raise ValueError('unknown model name `%s`' % model_name)

# # For now, we do not support relation_model
# relation_model_dict = [
#     'RSR',
#     'relation_GATs',
#     'relation_GATs_3heads'
# ]


class Benchmark:
    def __init__(self, data_dir="cn_data", market="csi300", model_type="linear", alpha=360,
                 lr=0.001, early_stop=10, horizon=1, rank_label=True,
                 h_path: Optional[str] = None,
                 train_start: Optional[str] = None,
                 test_start: Optional[str] = None,
                 test_end: Optional[str] = None, ) -> None:
        self.data_dir = data_dir
        self.market = market
        self.horizon = horizon
        self.model_type = model_type
        self.h_path = h_path
        self.train_start = train_start
        self.test_start = test_start
        self.test_end = test_end
        self.alpha = alpha
        self.rank_label = rank_label
        self.lr = lr
        self.early_stop = early_stop

    def basic_task(self):
        """For fast training rolling"""
        if self.model_type == "MLP":
            conf_path = (DIRNAME.parent / "benchmarks" / "MLP" / "workflow_config_mlp_Alpha{}.yaml".format(
                self.alpha))
            # filename = "MLP_alpha{}_handler_horizon{}.pkl".format(self.alpha, self.horizon)
        else:
            conf_path = (
                        DIRNAME.parent / "benchmarks" / self.model_type / "workflow_config_{}_Alpha{}.yaml".format(
                    self.model_type.lower(), self.alpha))
            # filename = "alpha{}_handler_horizon{}.pkl".format(self.alpha, self.horizon)

        # filename = f"{self.data_dir}_{self.market}_rank{self.rank_label}_{filename}"
        # h_path = DIRNAME.parent / "baseline" / filename
        # h_path = DIRNAME / filename
        # if self.h_path is not None:
        #     h_path = Path(self.h_path)

        with conf_path.open("r") as f:
            conf = yaml.safe_load(f)

        # modify dataset horizon
        conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["label"] = [
            "Ref($close, -{}) / Ref($close, -1) - 1".format(self.horizon + 1)
        ]

        if self.market != "csi300":
            conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"] = self.market
            if self.data_dir == "us_data":
                conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["label"] = [
                    "Ref($close, -{}) / $close - 1".format(self.horizon)
                ]

        batch_size = 5000
        # batch_size = 10
        for k, v in {'early_stop': self.early_stop, "batch_size": batch_size, "lr": self.lr, "seed": None, }.items():
            if k in conf["task"]["model"]["kwargs"]:
                conf["task"]["model"]["kwargs"][k] = v
        if conf["task"]["model"]["class"] == "TransformerModel":
            conf["task"]["model"]["kwargs"]["dim_feedforward"] = 32
            conf["task"]["model"]["kwargs"]["reg"] = 0

        task = conf["task"]

        # h_conf = task["dataset"]["kwargs"]["handler"]
        # if not h_path.exists():
        #     from qlib.utils import init_instance_by_config
        #     h = init_instance_by_config(h_conf)
        #     h.to_pickle(h_path, dump_all=True)
        #     print('Save handler file to', h_path)
        # task["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
        # print("task", task)

        if self.train_start is not None:
            seg = task["dataset"]["kwargs"]["segments"]["train"]
            task["dataset"]["kwargs"]["segments"]["train"] = pd.Timestamp(self.train_start), seg[1]

        if self.test_start is not None:
            seg = task["dataset"]["kwargs"]["segments"]["train"]
            task["dataset"]["kwargs"]["segments"]["test"] = pd.Timestamp(self.test_start), seg[1]

        if self.test_end is not None:
            seg = task["dataset"]["kwargs"]["segments"]["test"]
            task["dataset"]["kwargs"]["segments"]["test"] = seg[0], pd.Timestamp(self.test_end)


        return task


class IncrementalExp:
    """
    Example:
    python -u main.py run_all --forecast_model GRU -num_head 8 --tau 10 --first_order True --adapt_x True --adapt_y True --market csi300 --data_dir crowd_data --rank_label False
    """

    def __init__(
            self, args, data_dir="cn_data", root_path='../qlib_data', calendar_path=None, market="csi300",
            horizon=1, alpha=360, x_dim=None, step=20, # model_name="MLP",
            lr=0.01, lr_model=0.001, reg=0.5, num_head=8, tau=10, naive=False, preprocess_tensor=True,
            use_extra=False, tag=None, rank_label=False, first_order=True, h_path=None, test_start=None, test_end=None,
    ):
        """

        Parameters
        ----------
        data_dir (str):
            source data dictionary under root_path
        root_path (str):
            the root path of source data. '~/.qlib/qlib_data/' by default.
        root_path (str):
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
        model_name (str):
            consistent with directory name under examples/benchmarks
        lr (float):
            learning rate of data adapter
        lr_model (float):
            learning rate of forecast model and model adapter
        reg (float):
            regularization strength
        num_head (int):
            number of transformation heads
        tau (float):
            softmax temperature
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
        test_start (str):
            override the start date of test data
        test_end (str):
            override the end date of test data
        """
        self.data_dir = data_dir
        self.provider_uri = os.path.join(root_path, data_dir)

        if calendar_path is None:
            calendar_path = os.path.join(root_path, data_dir, 'calendars/day.txt')
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
        self.lr_model = lr_model
        self.num_head = num_head
        self.temperature = tau
        self.first_order = first_order
        self.naive = naive
        self.reg = reg
        self.not_sequence = self.model_name in ["MLP", 'Linear'] and self.alpha == 158

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

        if test_start is not None:
            self.segments['test'] = (test_start, self.segments['test'][1])
        if test_end is not None:
            self.segments['test'] = (self.segments['test'][0], test_end)

        # self.test_slice = slice(self.ta.align_time(test_start, tp_type='start'), self.ta.align_time(test_end, tp_type='end'))
        self.test_slice = slice(self.ta.align_time(self.segments['test'][0], tp_type='start'), self.ta.align_time(self.segments['test'][1], tp_type='end'))
        self.early_stop = args.early_stop
        self.h_path = h_path
        self.preprocess_tensor = preprocess_tensor
        self.use_extra = use_extra

        self.factor_num = 6 if self.alpha == 360 else 20
        self.x_dim = x_dim if x_dim else (360 if self.alpha == 360 else 20 * 20)
        # print('Experiment name:', self.experiment_name)

    @property
    def experiment_name(self):
        return f"{self.market}_{self.model_name}_alpha{self.alpha}_horizon{self.horizon}_step{self.step}" \
               f"_rank{self.rank_label}_{self.tag}"

    @property
    def basic_task(self):
        # this benchmark is important
        return Benchmark(
            data_dir=self.data_dir,
            market=self.market,
            model_type=self.model_name,
            horizon=self.horizon,
            rank_label=self.rank_label,
            alpha=self.alpha,
            lr=self.lr_model,
            early_stop=self.early_stop,
            h_path=self.h_path,
            test_start=self.test_slice.start,
            test_end=self.test_slice.stop,
            # train_start=self.segments['train'][0],
            # test_start=self.segments['test'][0],
            # test_end =self.segments['test'][1]
        ).basic_task()

    def _init_model(self, args):
        param_dict = json.load(open(args.model_path + '/info.json'))['config']
        param_dict['model_dir'] = args.model_path
        stock2concept_matrix = param_dict['stock2concept_matrix']
        stock2stock_matrix = param_dict['stock2stock_matrix']

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
        # elif param_dict['model_name'] == 'HIST':
        #     # HIST need stock2concept matrix, send it to device
        #     model = get_model(param_dict['model_name'])(d_feat=param_dict['d_feat'], num_layers=param_dict['num_layers']
        #                                                 , K=param_dict['K'])
        # elif param_dict['model_name'] in relation_model_dict:
        #     stock2stock_matrix = torch.Tensor(np.load(stock2stock_matrix)).to(args.device)
        #     num_relation = stock2stock_matrix.shape[2]  # the number of relations
        #     model = get_model(param_dict['model_name'])(num_relation=num_relation, d_feat=param_dict['d_feat'],
        #                                                 num_layers=param_dict['num_layers'])
        else:
            model = get_model(param_dict['model_name'])(d_feat=param_dict['d_feat'],
                                                        num_layers=param_dict['num_layers'])

        model.to(args.device)
        model.load_state_dict(torch.load(param_dict['model_dir'] + '/model.bin', map_location=args.device))
        # model = get_model('MLP')(d_feat=param_dict['d_feat'], num_layers=param_dict['num_layers'])
        # model.load_state_dict(torch.load(param_dict['model_dir'] + '/model.bin', map_location=args.device))
        return model

    def offline_training(self, args, segments: Dict[str, tuple] = None, data: pd.DataFrame = None, reload_path=None):
        model = self._init_model(args)
        model.to(args.device)

        if self.naive:
            framework = IncrementalManager(model, x_dim=self.x_dim, lr_model=self.lr_model, begin_valid_epoch=0, args = args)
        else:
            print("DoubleAdaptManager")
            framework = DoubleAdaptManager(model, x_dim=self.x_dim, lr_model=self.lr_model,
                                           first_order=True, begin_valid_epoch=0, factor_num=self.factor_num,
                                           lr_da=self.lr, lr_ma=self.lr_model,
                                           adapt_x=True, adapt_y=True, reg=self.reg,
                                           num_head=self.num_head, temperature=self.temperature, args = args)

        if reload_path is not None:
            framework.load_state_dict(torch.load(reload_path))
            print('Reload experiment', reload_path)
        else:
            if segments is None:
                segments = self.segments
            rolling_tasks = utils.organize_all_tasks(segments, self.ta, step=self.step, trunc_days=self.horizon + 1,
                                                     rtype=utils.TimeAdjuster.SHIFT_SD, use_extra=self.use_extra)
            rolling_tasks_data = {k: utils.get_rolling_data(rolling_tasks[k],
                                                            # data=self._load_data() if data is None else data,
                                                            data=data,
                                                            factor_num=self.factor_num, horizon=self.horizon,
                                                            not_sequence=self.not_sequence,
                                                            sequence_last_dim=self.alpha == 158,
                                                            to_tensor=self.preprocess_tensor)
                                  for k in ['train', 'valid']}
            framework.fit(meta_tasks_train = rolling_tasks_data['train'], meta_tasks_val = rolling_tasks_data['valid'], args = args)
        return framework

    def inference(self, args, model, meta_tasks_test: List[Dict[str, Union[pd.DataFrame, np.ndarray, torch.Tensor]]],
                  date_slice: slice = slice(None, None), stock2concept_matrix=None, stock2stock_matrix=None):
        param_dict = json.load(open(args.model_path + '/info.json'))['config']
        param_dict['model_dir'] = args.model_path
        stock2concept_matrix = param_dict['stock2concept_matrix']
        stock2stock_matrix = param_dict['stock2stock_matrix']
        model.eval()

        pred_y_all, mse_all = [], 0
        indices = np.arange(len(meta_tasks_test))
        for i in tqdm(indices, desc="online") if True else indices:
            meta_input = meta_tasks_test[i]
            if not isinstance(meta_input['X_train'], torch.Tensor):
                meta_input = {
                    k: torch.tensor(v, device=args.device, dtype=torch.float32) if 'idx' not in k else v
                    for k, v in meta_input.items()
                }

            """ Online inference """
            if "X_extra" in meta_input and meta_input["X_extra"].shape[0] > 0:
                X_test = torch.cat([meta_input["X_extra"].to(args.device), meta_input["X_test"].to(args.device), ], 0, )
                y_test = torch.cat([meta_input["y_extra"].to(args.device), meta_input["y_test"].to(args.device), ], 0, )
            else:
                X_test = meta_input["X_test"].to(args.device)
                y_test = meta_input["y_test"].to(args.device)
            if X_test.dim() == 3:
                X_test = X_test.permute(0, 2, 1).reshape(len(X_test), -1) if True else X_test.reshape(len(X_test), -1)

            stock_index = None  # need to be revise
            market_value = None
            with torch.no_grad():
                if args.model_name == 'HIST':
                    pred = model(X_test, stock2concept_matrix[stock_index], market_value)
                # elif args.model_name in relation_model_dict:
                #     pred = model(X_test, stock2stock_matrix[stock_index][:, stock_index])
                else:
                    pred = model(X_test)
                    pred = pred.view(-1)


            test_begin = len(meta_input["y_extra"]) if "y_extra" in meta_input else 0
            output = pred[test_begin:].detach().cpu().numpy()

            test_idx = meta_input["test_idx"]
            pred_y_all.append(
                pd.DataFrame(
                    {
                        "pred": pd.Series(output, index=test_idx),
                        "label": pd.Series(meta_input["y_test"], index=test_idx),
                    }
                )
            )
        pred_y_all = pd.concat(pred_y_all)
        pred_y_all = pred_y_all.loc[date_slice]
        return pred_y_all

    def online_training(self, args, segments: Dict[str, tuple] = None,
                        data: pd.DataFrame = None, reload_path: str = None, framework=None, ):
        """
        Perform incremental learning on the test data.

        Parameters
        ----------
        segments (Dict[str, tuple]):
            The date range of training data, validation data, and test data.
            Example:
                {
                    'train': ('2008-01-01', '2014-12-31'),
                    'valid': ('2015-01-01', '2016-12-31'),
                    'test': ('2017-01-01', '2020-08-01')
                }
        data (pd.DataFrame):
            the index col is pd.MultiIndex with the datetime as level 0 and the stock ID as level 1;
            the col named 'feature' contains the stock feature vectors;
            the col named 'label' contains the ground-truth labels.
        reload_path (str):
            if not None, reload checkpoints

        Returns
        -------
        pd.DataFrame:
            the index col is pd.MultiIndex with the datetime as level 0 and the stock ID as level 1;
            the col named 'pred' contains the predictions of the model;
            the col named 'label' contains the ground-truth labels which have been preprocessed and may not be the raw.
            :param reload_path:
            :param data:

        """
        if framework is None:
            model = self._init_model()
            model.to(args.device)
            if self.naive:
                framework = IncrementalManager(model, x_dim=self.x_dim, lr_model=self.lr_model,
                                               first_order=True, alpha=self.alpha, begin_valid_epoch=0)
            else:
                framework = DoubleAdaptManager(model, x_dim=self.x_dim, lr_model=self.lr_model,
                                               first_order=True, begin_valid_epoch=0, factor_num=self.factor_num,
                                               lr_da=self.lr, lr_ma=self.lr_model,
                                               adapt_x=True, adapt_y=True, reg=self.reg,
                                               num_head=self.num_head, temperature=self.temperature)
            if reload_path is not None:
                framework.load_state_dict(torch.load(reload_path))
                print('Reload experiment', reload_path)
        else:
            model = self._init_model(args)
            model.to(args.device)

        if segments is None:
            segments = self.segments
        rolling_tasks = utils.organize_all_tasks(segments, self.ta, step=self.step, trunc_days=self.horizon + 1,
                                                 rtype=utils.TimeAdjuster.SHIFT_SD, use_extra=self.use_extra)
        rolling_tasks_data = utils.get_rolling_data(rolling_tasks['test'],
                                                    # data=self._load_data() if data is None else data,
                                                    data=data,
                                                    factor_num=self.factor_num, horizon=self.horizon,
                                                    not_sequence=self.not_sequence,
                                                    sequence_last_dim=self.alpha == 158,
                                                    to_tensor=self.preprocess_tensor)

        return framework.inference(meta_tasks_test=rolling_tasks_data, date_slice=self.test_slice, args=args), \
               self.inference(model=model, args=args, meta_tasks_test=rolling_tasks_data, date_slice=self.test_slice)

    def _evaluate_metrics(self, pred: pd.DataFrame, name):
        from qlib.utils import init_instance_by_config
        from qlib.data.dataset import DataHandlerLP

        """        
        Note that the labels in pred_y_all are preprocessed. IC should be calculated by the raw labels. 
        """

        ds = init_instance_by_config(self.basic_task["dataset"])
        # print("self.basic_task[dataset]", self.basic_task["dataset"])
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

    def workflow(self, args, reload_path: str = None):
        if args.model_save_path:
            if not os.path.exists(args.model_save_path):
                os.mkdir(args.model_save_path)
            # save_path = os.path.join(args.model_save_path, f"{self.experiment_name}.pt")

        data = create_doubleadapt_loaders(args)

        print(self.segments)
        assert data.index[0][0] <= self.ta.align_time(self.segments['train'][0], tp_type='start')
        assert data.index[-1][0] >= self.ta.align_time(self.segments['test'][-1], tp_type='end')
        print("offline_training")
        framework = self.offline_training(args=args, data=data, reload_path=reload_path)

        print("evaluation")
        pred_y_all_incre, pred_y_all_basic = self.online_training(data=data, framework=framework, args=args,
                                                                  reload_path=reload_path)
        self._evaluate_metrics(pred_y_all_incre, "incre_model")
        self._evaluate_metrics(pred_y_all_basic, "basic_model")
        if reload_path is not None:
            pd.to_pickle(pred_y_all_incre, args.result_path)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model_name', default='GATs')
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--early_stop', type=int, default=200)
    parser.add_argument('--reload_path', default="./output/for_platform/INCRE/DA_GATs/model.bin",
                        help='None: without reloading; str: path for loading')
    parser.add_argument('--result_path', default="./pred_output/GATs_DA.pkl")
    parser.add_argument('--model_path', default='./output/for_platform/GATs', help='learned model')
    parser.add_argument('--model_save_path', default="./output/for_platform/INCRE/DA_GATs", help='updated model')
    parser.add_argument('--incre_train_start', default='2008-01-01')  # work time 2012 0101
    parser.add_argument('--incre_train_end', default='2018-12-31')  # work time 2012 1231
    parser.add_argument('--incre_val_start', default='2019-01-01')  # work time 2013 0101
    parser.add_argument('--incre_val_end', default='2019-12-31')  # work time 2013 1231
    parser.add_argument('--test_start', default='2020-01-01')  # work time 2014 0101
    parser.add_argument('--test_end', default='2023-06-28')  # work time 2014 1231

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    a = IncrementalExp(args=args)
    a.workflow(args=args, reload_path=args.reload_path)

