# TE-DyGE: Temporal Evolution-enhanced Dynamic Graph Embedding Network
##### Contributors: Liane WANG
> Liping Wang, Yanyan Shen, and Lei Chen. 2023. TE-DyGE: Temporal Evolution-Enhanced Dynamic Graph Embedding Network. In Database Systems for Advanced Applications: 28th International Conference, DASFAA 2023, Tianjin, China, April 17–20, 2023, Proceedings, Part III. Springer-Verlag, Berlin, Heidelberg, 183–198. https://doi.org/10.1007/978-3-031-30675-4_13

Please cite our papers:
@inproceedings{10.1007/978-3-031-30675-4_13,
author = {Wang, Liping and Shen, Yanyan and Chen, Lei},
title = {TE-DyGE: Temporal Evolution-Enhanced Dynamic Graph Embedding Network},
year = {2023},
url = {https://doi.org/10.1007/978-3-031-30675-4_13},
booktitle = {Database Systems for Advanced Applications: 28th International Conference, DASFAA 2023, Tianjin, China, April 17–20, 2023, Proceedings, Part III},
pages = {183–198},
numpages = {16},
keywords = {Representation Learning, Dynamic Graphs},
}

## Framework

![TE-DyGE: Temporal Evolution-enhanced Dynamic Graph Embedding Network](tedyge.png)

## Implementation
[TE-DyGE](https://github.com/liane886/TE-DyGE)

## Setup
```
conda env create -f TE_DyGE.yml
```
## Example Usage
To reproduce the experiments on EComm dataset, simply run:
```
python run_script.py
```
## Dataset 
Statistics of dataset:

![dataset](data/Dataset.png)

## Results
Results for each snapshpot can be found in the 'logs' file.

## Acknowledgement
The original version of this code base was originally forked from [DySAT] https://github.com/aravindsankar28/DySAT 
```
@inproceedings{Xue2020DyHATR,
  title     = {Modeling Dynamic Heterogeneous Network forLink Prediction using Hierarchical Attentionwith Temporal RNN},
  author    = {Xue, Hansheng and Yang, Luwei and Jiang, Wen and Wei, Yi and Hu, Yi and Lin, Yu},
  booktitle = {Proceedings of the 2020 European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD)},
  year      = {2020},
}
```
