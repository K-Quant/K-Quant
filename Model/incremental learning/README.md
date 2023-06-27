# Incremental Learning


## 0. run

```shell
python incremental.py \
    --model_name 'MLP' \
    --train_start_date '2008-01-01' \
    --train_end_date '2013-12-31' \
    --incremental_start_date '2014-01-01'\
    --incremental_end_date '2014-12-31'\
    --baseline_start_date '2008-01-01'\
    --baseline_end_date '2014-12-31'\
    --val_start_date '2015-01-01'\
    --val_end_date '2015-12-31'\
    --test_start_date '2016-01-01'\
    --test_end_date '2018-12-31'\
    --n_epochs 1\
    --incre_n_epochs 1
```


- ``model_name``: could choose from: 'MLP', 'LSTM', 'ALSTM', 'GRU', 'GAT', 'HIST', 'RSR'
- ``n_epochs``: The number of epochs of the baseline model and the basic model.
- ``incre_n_epochs``: The number of epochs of the incremental learning.

## 1. result
When n_epoch = 1, incre_n_epochs = 1

|               | IC | Rank_IC | Time |
| ------------- | ------- ------ |
| MLP  | 26.339%  | 31.134%	| 61.247% |
|ALSTM	|0.282%	|1.794%	| 65.273%|
|LSTM	|8.626%	|12.056%	|60.727%|
|GRU	|0.102%	|0.506%	|59.976%|
|GAT	|-1.291%	|-1.172%	|65.574%|
|HIST	|-1.406%	|-2.927%	|68.422%|
|RSR	|-2.164%	|-3.495%	|62.439%|

When n_epoch = 200, incre_n_epochs = 200

|               | IC | Rank_IC | Time |
| ------------- | ------- ------ |
|MLP	|2.884%	|9.286%	|58.665%
|ALSTM	|32.243%	|34.511%	|67.009%
|LSTM	|28.459%	|22.067%	|62.765%
|GRU	|12.182%	|9.845%	|62.684%
|GAT	|8.245%	|3.775%	|65.097%
|HIST	|57.996%	|57.324%	|68.197%
|RSR	|3.691%	|1.251%	|64.222%


