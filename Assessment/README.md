# Comprehensive Evaluation System for Multi-dimensional Intelligent Quantitative Investment Models.. 


## Introduction

This system consists of two parts:
1. Multi-dimensional performance evaluation for different combinations of forecasting and explanatory models.
2. Performance assessment of different investment portfolios.

In the first part, users can select different forecasting and explanatory models as well as examination periods. The evaluation system will output comprehensive performance scores for different combinations.Additionally, the forecasting models in the first part will also provide recommendations of three stocks for users to consider.
In the second part, users can input different investment portfolios based on their own stock selection results, and test the performance of these portfolios over different time periods.



## Environment
1. Install python3.8(recommend) 
2. Install the requirements by the following guide.
```
# install pyg from source
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+${CUDA}.html

# install dgl from source
conda install -c dglteam/label/cu121 dgl

# install tushare from source
pip install tushare
```
3. Download the stocks related data from [this link](https://drive.google.com/file/d/1v_DkQZN6aWEAeZkZRbc1WvJ_1xi9vhzP/view?usp=sharing) and put them into K-Quant/Data;
4. Download the assessment related data from [this link](https://drive.google.com/file/d/10rGJRElRzh6-tS7R_wM8CDiBc7XHdMTl/view?usp=sharing) and put them into K-Quant/Data;
5. To get the up-to-date time series data, we recommend using the following Qlib Alpha 360 data source:
```commandline
wget https://github.com/chenditc/investment_data/releases/download/2023-07-01/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C cn_data --strip-components=2
```
7. This Jupyter notebook [Assessment_example.ipynb](../Assessment_example.ipynb) notebook illustrates how to use explanation models
## Acknowledgement

Thanks to research work [HIST](https://github.com/Wentao-Xu/HIST) and [Time-Series-Library](https://github.com/thuml/Time-Series-Library/)
