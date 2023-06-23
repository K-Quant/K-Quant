# -*- coding:utf-8 -*-
import codecs
import json
import os

dirname, filename = os.path.split(os.path.abspath(__file__))


def get_pose(text, E_name):
    result = []
    pose_1 = text.find(E_name)
    pose_2 = pose_1 + len(E_name) - 1
    result.append(pose_1)
    result.append(pose_2)
    return result


# 处理后数据,处理完了再分训练集和验证集
f = open(dirname + '/data/comp_all.txt', "w+")
# f1 = open('comp_train.txt', "w+")
# f2 = open('comp_test.txt', 'w+')
# 原始数据,数据来源：https://github.com/buppt/ChineseNRE
company_relation = dirname + "/data/excel2txt.txt"

rel_set = []
with codecs.open(company_relation, 'r', 'utf-8') as tfc:
    for lines in tfc:
        line = lines.split()
        data = {}
        if len(line) != 4:
            continue
        # E1, E2, relation, text = line
        print(line)
        E1, E2, text = line
        data["token"] = list(text)
        h = {}
        h["name"] = E1
        h["pos"] = get_pose(text, E1)
        data["h"] = h
        t = {}
        t["name"] = E2
        t["pos"] = get_pose(text, E2)
        data["t"] = t
        data["relation"] = "合作"
        rel_set.append(data)

    rel_test = rel_set[int(len(rel_set)):]

    for x in rel_test:
        json_data = json.dumps(x, ensure_ascii=False)
        f.write(json_data)
        f.write("\n")
