# Stock prediction module


## Introduction

This repo is for K-Quant project, stock forecasting module.

This module have 3 basic functions:
### Module 2.1 knowledge based stock recommendation models
### Module 2.2 stock recommendation ensemble models
### Module 2.3 incremental learning methods for stock models

## Environment
1. Install python3.8(recommend) 
2. Install the requirements in [requirements.txt].
3. Install the quantitative investment platform [Qlib](https://github.com/microsoft/qlib) and download the data from Qlib:
    ```
    # install Qlib from source
    pip install --upgrade  cython
    git clone https://github.com/microsoft/qlib.git && cd qlib
    python setup.py install

    # Download the stock features of Alpha360 from Qlib
    # the target_dir is the same as provider_url in utils/dataloader.py
    python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn --version v2
    ```
4. Download market_value, index file for knowledge empowered models from [this link](https://drive.google.com/file/d/1KBwZ_lX___bYBIHx9VWRzRgLFb8N3-NK/view?usp=sharing)
5. To get the up-to-date time series data, we recommend using the following Qlib Alpha 360 data source:
```commandline
wget https://github.com/chenditc/investment_data/releases/download/2023-07-01/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C cn_data --strip-components=2
```

## Module 2.1 stock recommendation models
Now we provide the following models that could be used in stock regression/forecasting/recommendation:
```
------------------basic deep learning models------------
MLP
GRU
LSTM
ALSTM
SFM
GATs
------------------models powered by knowledge-----------
HIST
RSR
relation_GATs
KEnhance
------models that SOTA on other time series library-----
------this part is under finetune-----------------------
DLinear [AAAI 2023]
Autoformer [NeurIPS 2023]
Crossformer [ICLR 2023]
ETSformer
FEDformer [ICML 2022]
FiLM [NeurIPS 2022]
Informer [AAAI 2021]
PatchTST [ICLR 2023]
---------------------------------------------------------
```
### Run experiments
    python learn.py --model_name [model you choose] --outdir 'output/[folder your named]'
### Results
The result will be stored in output folder, if you need some well-trained models, we provide in [this link](https://drive.google.com/file/d/1yGHXZDcCgY4AAp_UM_gKXyKo25Atmoft/view?usp=sharing)
### Knowledge choice
For models in relation_model_dict(in exp/learn.py), different knowledge source could be chosen as the knowledge input, we have the following choice:
```angular2html
industry-relation
hidy-relation[extracted from HiDy in Module 1]
dueefin
shanghai tech
Fr2kg
Doc2edga
```


### To save the prediction result:
modify the ```prefix``` and ```model_pool``` in ```exp/ensemble_basic.py```.

Then run ```batch_prediction``` in ```exp/ensemble_basic.py```.

You can get multi models prediction results in one pickle file.

### Backtest

To run the backtest to evaluate the model performances on investment, 
run ```backtest.py``` to get the report or figure of cumulated excess return.

The backtest need the prediction result from ```exp/ensemble_basic.py```

### Attention:
For knowledge empowered model, we only support use THE SAME file while you train the model

So when it comes to dynamic knowledge, you need to update the knowledge file and cover the path in ```exp/prediction.py main()```

For example, HIST needs up-to-date market value, and we use old one now which may could impact the model
s performance.

## Module 2.2 Stock ensemble models

In this module, we provide several ensemble methods:
```angular2html
average ensemble
linear blend ensemble
dynamic linear blend ensemble
performance based ensemble
Rensemble no retrain
Rensemble with retrain
```
To get the result of the average and linear blend ensemble: run ```average_and_blend``` in ```exp/ensemble_basic.py```

To get the result of the dynamic linear blend ensemble: run ```sim_linear``` in ```exp/ensemble_basic.py```

To get the result of performance based ensemble, rensemble with/without retrain: run  ```ensemble_sjtu``` in ```exp/ensemble_basic.py```

The ensemble result will be stored in the pickle file like module 2.1.

The ensemble model could also be evaluated with backtest, the same as module 2.1.

## Module 2.3 Incremental learning models

Here we provide two different incremental learning methods. ```gradient based incremental learning``` and ```DoubleAdapt```

To use ```gradient based incremental learning```, run ```exp/learn_incre.py```, and the model will be saved in the path you choose.

The evaluation of gradient based incre model is the same as models in module 2.1, only modified the incremental path and enable the incremental mode.

To use ```DoubleAdapt```, run ```exp/learn_incre_DoubleAdapt.py``` with ```reload_path``` set to ```None```

To evaluation ```DoubleAdapt```, run ```exp/learn_incre_DoubleAdapt.py``` with ```reload_path``` set to a saved DoubleAdapt model
## Acknowledgement

Thanks to research work [HIST](https://github.com/Wentao-Xu/HIST) and [Time-Series-Library](https://github.com/thuml/Time-Series-Library/)
