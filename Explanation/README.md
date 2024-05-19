# Explanation Methods for Knowledge 


## Introduction

This repo is for K-Quant project, explanation module.

This module has 4 basic explanation models and only knowledge-based models can be explained:
### Models 1  Input-Gradient Explainer
### Models 2  Xpath Explainer
### Models 3  Gnn Explainer
### Models 4  Hencex Explainer

## Environment
1. Install python3.8(recommend) 
2. Install the requirements by the following guide.
```
# install pyg from source
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+${CUDA}.html

# install dgl from source
conda install -c dglteam/label/cu121 dgl
```
3. Download the data from [this link](https://drive.google.com/file/d/1v_DkQZN6aWEAeZkZRbc1WvJ_1xi9vhzP/view?usp=sharing) and put them into K-Quant/Data:
4. To get the up-to-date time series data, we recommend using the following Qlib Alpha 360 data source:
```commandline
wget https://github.com/chenditc/investment_data/releases/download/2023-07-01/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C cn_data --strip-components=2
```
5.  This Jupyter notebook [Explanation_example.ipynb](../Explanation_example.ipynb) notebook illustrates how to use explanation models
## Acknowledgement

Thanks to research work [HIST](https://github.com/Wentao-Xu/HIST) and [Time-Series-Library](https://github.com/thuml/Time-Series-Library/)
