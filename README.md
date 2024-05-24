# K-Quant: A Platform of Temporal Financial Knowledge-enhanced Quantitative Investment

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.html)
![](https://img.shields.io/badge/platform-win%20%7C%20linux-yellow.svg)
![](https://img.shields.io/badge/python--language-2.7-red.svg)
[![K-Quant Website](https://img.shields.io/website-up-down-green-red/https/shields.io.svg?label=K-Quant%20website)](http://lccpu22.cse.ust.hk:8003/new/index.html#/welcome)
[![Survey](https://img.shields.io/badge/Survey%20%E2%AD%90-%23FF8000)](https://arxiv.org/abs/2308.04947)

## Overview
In recent years, domain-specific knowledge bases (KBs) have attracted more attention in academics and industry because of their expertise and in-depth representation in a specific domain. However, when constructing a domain-specific KB, one needs to address not only the challenges in constructing a general KB, but also the difficulties raised by the nature of domain-specific raw data. Considering the usability of financial Knowledge Bases (KBs) in many downstream applications, such as financial risk analysis and fraud detection, we propose a Platform for Temporal Financial KB Construction, K-Quant Platform. 

K-Quant system have three main components: Knowledge Module, Quantitative Investment Module, XAI Module.

## Main Features
![overview](logo/demo_overview.png)
The main features of K-Quant:

#### 1. Dynamic Financial KB Construction

The dynamic financial KB construction module consists of three main steps:

1. Evolved Knowledge Extraction: it extracts financial knowledge with evolved information from several reliable data sources.

2. Temporal Record Linkage and Conflict Resolution: it removes duplicated and conflicted knowledge based on temporal information to increase the confidence of the delivered knowledge.

3. Dynamic Knowledge Update: it proposes a temporal pattern-based inference rule learning module to maintain our constructed KB consistent with the dynamically changing world.

#### 2. Quantitative Investment Framework
To handle temporal relations between companies and utilize the knowledge in quantitative investment, the K-Quant have three main functions in Quantitative Investment Framework:
  1. Knowledge based stock recommendation models

     We build knowledge based deep learning models for stock forecasting tasks. By extracting knowledge formed in KB construction module as relational matrix, we use them as an alternative data. For detailed models, please refer to [readme of stock prediction](Model/model_pool/README.md)
  2. Stock recommendation ensemble models 

     To boost the performance of stock recommendation, we employ ensemble methods to sample the output of different deep learning models. To better capture the rapid change of stock prices, we design several dynamic ensemble methods including **Online ensemble method based on resampling distribution estimation** and **dynamically heterogeneous ensemble methods**. Both of these two methods apply for patents.
  3. Incremental learning methods for stock models

     The complex stock market make deep learning model invalid soon because of the distribution shift. We propose incremental learning methods to mitigate the influence of such shift.

#### 3. Evaluation and Interpretability Methods

<!-- TOC -->

## Outline

- [Overview](#overview)
- [Main Features](#main-features)
- [Outline](#outline)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Benchmark Dataset](#benchmark-dataset)
  - [Description](#description)
  - [Data Format](#data-format)
- [Related Work](#related-work)
- [Acknowledgement](#acknowledgement)

<!-- /TOC -->

## Quick Start

### Prerequisites

### Installation
For Quantitative Investment Framework, you need install [Microsoft Qlib](https://github.com/microsoft/qlib) 
and download up-to-date stock time series data.

We recommend install Qlib from source and download the crowd source stock data
    
```
# install Qlib from source
pip install --upgrade  cython
git clone https://github.com/microsoft/qlib.git && cd qlib
python setup.py install
# Download the stock features of Alpha360 from Qlib
# the target_dir is the same as provider_url in utils/dataloader.py
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn --version v2    
```
### Usage
For detailed usage of quantitative forecasting model, please refer to 
[Stock prediction module](Model/model_pool/README.md)

## Benchmark Dataset
Please refer to our [HiDy Dataset](https://github.com/K-Quant/HiDy).

### Description
HiDy is a hierarchical, dynamic, robust, diverse, and large-scale financial benchmark KB that aims to provide various valuable financial knowledge as critical benchmarking data for fair model testing in different financial tasks. Specifically, HiDy currently contains 34 relation types and 17 entity types. The scale of HiDy is steadily growing due to its continuous updates. To make HiDy easily accessible and retrieved, HiDy is organized in a well-formed financial hierarchy with four branches, Macro, Meso, Micro, and Others.


### Data Format

Each instance in the dataset is in the form of quadruple (entity 1, relation , entity 2, timestamp).


## Published Papers

If you find this system helpful in your research, please consider citing our papers that are listed below:

> Cost-aware Outdated Facts Correction in the Knowledge Bases. Hao Xin, Yanyan Shen, Lei Chen. DASFAA 2024. [Paper Coming Soon]

> PKBC: A Product-Specific Knowledge base Taxonomy Framework. Hao Xin, Yanyan Shen, Lei Chen. DASFAA 2024. [Paper Coming Soon]

> Triple-d: Denoising Distant Supervision for High-quality Data Creation. Xinyi Zhu, Yongqi Zhang, Lei Chen, Kai Chen. ICDE 2024. [Paper Coming Soon]

> T-FinKB: A Platform of Temporal Financial Knowledge Base Construction. Xinyi Zhu, Liping Wang, Hao Xin, Xiaohan Wang, Zhifeng Jia, Jiyao Wang, Chunming Ma, Yuxiang Zeng. ICDE 2023. [Paper Coming Soon]

> Liping Wang, Yanyan Shen, and Lei Chen. 2023. TE-DyGE: Temporal Evolution-Enhanced Dynamic Graph Embedding Network. In Database Systems for Advanced Applications: 28th International Conference, DASFAA 2023, Tianjin, China, April 17–20, 2023, Proceedings, Part III. Springer-Verlag, Berlin, Heidelberg, 183–198. https://doi.org/10.1007/978-3-031-30675-4_13

> Xinyi Zhu, Hao Xin, Yanyan Shen, and Lei Chen. 
HIT - An Effective Approach to Build a Dynamic Financial Knowledge Base. In Database Systems for Advanced Applications: 28th International Confer- ence, DASFAA 2023, Tianjin, China, April 17–20, 2023, Proceedings, Part II, pages 716–731. Springer, 2023. https://link.springer.com/chapter/10.1007/978-3-031-30672-3_48

> Liping Wang, Jiawei Li, Lifan Zhao, Zhizhuo Kou, Xiaohan Wang, Xinyi Zhu, Hao Wang, Yanyan Shen, Lei Chen.
Methods for Acquiring and Incorporating Knowledge into Stock Price Prediction: A Survey. https://arxiv.org/abs/2308.04947


