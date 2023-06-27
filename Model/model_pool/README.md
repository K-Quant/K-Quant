# Stock prediction module
This repo is for K-Quant project.
Now we provide the following models that could be used in stock regression/forecasting/recommendation:
```
-----------------------basic models--------------------
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
Triple_Att
------models that SOTA on other time series library-----
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
## Run experiments
    python learn.py --model_name [model you choose] --outdir 'output/[folder your named]'
## Results
The result will be stored in output folder, if you need some well-trained models, we provide in [this link](https://drive.google.com/file/d/1yGHXZDcCgY4AAp_UM_gKXyKo25Atmoft/view?usp=sharing)

## acknowledgement

Thanks to research work [HIST](https://github.com/Wentao-Xu/HIST) and [Time-Series-Library](https://github.com/thuml/Time-Series-Library/)