import json
import torch
import os
import opennre
from opennre import encoder, model, framework
import argparse

dicct = {'1': '增持',
         '2': '被增持',
         '3': '减持',
         '4': '被减持',
         '5': '供应',
         '6': '被供应',
         '7': '投资',
         '8': '被投资',
         '9': '上级',
         '10': '下级',
         '11': '同跌',
         '12': '同涨',
         '13': '合作',
         '14': '竞争',
         '15': '同行',
         '16': '纠纷',
         '0': '未知Unknown'
         }

dirname, filename = os.path.split(os.path.abspath(__file__))


def prediction_mode():
    # Some basic settings
    ckpt = dirname + '/ckpt/compay_rel_bert_softmax.pth.tar'

    # Check data
    rel2id = json.load(open(dirname + '/data/comp_rel2id.json'))

    # Define the sentence encoder
    sentence_encoder = opennre.encoder.BERTEncoder(
        max_length=80,
        pretrain_path=dirname + '/pretrain/chinese_wwm_pytorch',
        mask_entity='store_true'
    )

    # Define the model
    model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
    # model =model.cuda()
    # Define the whole training framework
    framework = opennre.framework.SentenceRE(
        train_path=None,
        val_path=None,
        test_path=dirname + '/data/comp_pred.txt',
        model=model,
        ckpt=ckpt,
        batch_size=32,  # Modify the batch size w.r.t. your device
        max_epoch=10,
        lr=2e-5,
        opt='adamw'
    )

    # TODO 如果不用cpu的话，把map_location删掉
    framework.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
    pred_result = framework.pred_label(framework.test_loader)
    for pred in pred_result:
        print(dicct[str(pred)])
    # Save the prediction
    with open(dirname + "/data/pre_result.json", 'w') as f:
        json.dump(pred_result, f)

# prediction_mode()
