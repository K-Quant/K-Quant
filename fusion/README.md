# Dynamic Knowledge Fusion

This task supports dynamic knowledge fusion with the input of quadruple, (head, tail, relation, timestamp), from the extraction module.

## 0. The directory structure of the code

```shell
.
├── K-Quant           # This is our project
    ├── extraction   # This is the company relation extraction folder
    ├── fusion         # This is this fusion folder
        ├── dart_pipeline.py      # This is the fusion pipeline.
        ├── DART.py
        ├── dynamic_pipline.py           # This is the demo of extraction and fusion pipeline.
        ├── jaro_winkler.py
        └── README.md
    ├── ...
```

## 1. Directly run extraction and fusion pipeline

```bash
python dynamic_pipline.py \
    --original_news_path ../Data/news.json \
    --extracted_data_path ../Data/data_extraction.json \
    --fusion_data_path ../Data/fusion.csv \
    --input_format "json"
```

