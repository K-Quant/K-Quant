# Dynamic Company Relation Extraction

This part includes knowledge in $Meso$, $Micro$ hierarchy. The extraction schema is defined as (company1, relation, company2, timestamp). For example, (Xiaomi, compete, Huawei, 2023.02.05).

## 0. The directory structure of the code

```shell
.
├── OpenNRE         # Please download and install the OpenNRE project and move the folder to this location
└── HiDy           # This is our project
    ├── company_rel   # This is the company relation extraction folder
        ├── data         # Dataset
        ├── pretrain      # Please download and move the folder to this location
            ├── ckpt3      # Please download and move the folder to this location
            ├── ckpt2      # Please download and move the folder to this location
            ├── ckpt      # Please download and move the folder to this location
        ├── Parsley      # Please download and install the Parsley project and move the folder to this location
        ├── __init__.py
        ├── company_rel_extraction.py
        ├── data_processing.py
        ├── gen.py
        ├── get_sen_set.py
        ├── demo.py           # Demo of relation extraction
        ├── dynamic_update_company_relation.py    # Extraction example
        └── README.md
    ├── ...
```


## 1. Download and install OpenNRE Project:

```bash
git clone https://github.com/thunlp/OpenNRE.git --depth 1
```

Move the project folder to the position as shown in the code structure.

And then add the follow function to class SentenceRE(nn.Module)

```python
def pred_label(self, eval_loader):
    pred_result = []
    with torch.no_grad():
        t = tqdm(eval_loader)
        for iter, data in enumerate(t):
            if torch.cuda.is_available():
                for i in range(len(data)):
                    try:
                        data[i] = data[i].cuda()
                    except:
                        pass
            label = data[0]
            args = data[1:]        
            logits = self.parallel_model(*args)
            score, pred = logits.max(-1) # (B)
            # Save result
            for i in range(pred.size(0)):
                pred_result.append(pred[i].item())
    return pred_result
```

Then modify the original warm up condition in SentenceRE.__init__ to:

```python
if warmup_step > 0 and train_path != None:
    from transformers import get_linear_schedule_with_warmup
    training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch
    self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step, num_training_steps=training_steps)
else:
    self.scheduler = None
```

Finally, set opennre as the development mode.

```bash
pip install -r requirements.txt
python setup.py develop
```


## 2. Download and install the Parsley project

```bash
git clone https://github.com/buckyroberts/Parsley.git
```
## 3. Download the pretrained model 
Download the pretrained models from
https://drive.google.com/drive/folders/1KjvVLkzgJHgqn_N1toOK_LMEEAYBhNc3

And move the models to the location that the directory structure of the code shows.

## 3. Do company relation extraction

```bash
python demo.py \
    --original_news_path ./data/news.json \
    --extracted_data_path ./data/data_extraction.json \
    --input_format "json"
```

