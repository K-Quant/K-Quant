#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-08-22 20:07
# @Author  : hxinaa
from sys import argv
import logging
import json
import itertools
import numpy as np
import pandas as pd
from score_functions import score_12

import rule_application as ra
from grapher import Grapher
from temporal_walk import Temporal_Walk
import os
import pickle
from tqdm import tqdm

config = json.load(open("./config.json"))
data_dir = config['data_dir']

redo = True
top_k = 10

def store_neighbors(quads):
    """
    Store all neighbors (outgoing edges) for each node.
    :param quads: indices of quadruples
    :return: a dict neighbors for each node
    """
    neighbors = dict()
    nodes = list(set(quads[:, 0]))
    for node in nodes:
        neighbors[node] = quads[quads[:, 0] == node]

    return neighbors


# if not os.path.exists(data_dir+'ts2id.json') or redo:
#     preprocess_graph(data_graph_file)

data = Grapher(data_dir)
graph = data.all_idx

if not os.path.exists(data_dir+'neighbors.pkl') or redo:
    neighbors = store_neighbors(graph)
    with open(data_dir + "neighbors.pkl", "wb") as fout:
        pickle.dump(neighbors, fout)
else:
    neighbors = pickle.load(open(data_dir + 'neighbors.pkl', "rb"))

def get_neighbors(fact,neighbors):
    """
    :param fact: extracted fact, fact = [s,p,o,t]
    :param neighbors: a dict stores all neighbors information, neighbors[x] = [f1,f2...,fn], f1=[s,p,o,t]
    :return: all the facts need to check
    """
    ans = list(neighbors[fact[0]])
    for x in neighbors[fact[2]]:
        ans.append(x)
    return ans

# def get_all_possible_facts(fact,data):
#     """
#     :param fact: the new extracted fact
#     :param data: graph data
#     :return:
#     """
#     ans = []
#     gth = {}
#     tid = 100
    
#     sid = int(fact[0])
#     gth[sid] = {}
#     for x in range(int(len(data.id2relation.keys())/2)):
#         ans.append([sid,x,0,tid])
#     for x in neighbors[sid]:
#         if x[1] not in gth[sid]:
#             gth[sid][x[1]] = []
#         gth[sid][x[1]].append([x[2], x[3]])

    
#     oid = int(fact[2])
#     gth[oid] = {}
#     for x in range(int(len(data.id2relation.keys())/2)):
#         ans.append([oid, x, 0, tid])
#     for x in neighbors[oid]:
#         if x[1] not in gth[sid]:
#             gth[sid][x[1]] = []
#         gth[sid][x[1]].append([x[2], x[3]])
#     return ans,gth

def get_all_possible_facts(factid,data):
    """
    :param fact: the new extracted fact in id format
    :param data: graph data
    :return: possible new facts, neighbors facts
    """
    ans = []
    gth = {}
    sid = factid[0]
    oid = factid[2]
    tid = 100
    
    for i in range(int(len(data.id2relation.keys())/2)):
        ans.append([sid,i,0,tid])
    for i in range(int(len(data.id2relation.keys())/2)):
        ans.append([oid,i,0,tid])
    
    gth[sid] = {}
    for x in neighbors[sid]:
        if x[1] not in gth[sid]:
            gth[sid][x[1]] = []
        gth[sid][x[1]].append([x[2], x[3]])
   
    gth[oid] = {}
    for x in neighbors[oid]:
        if x[1] not in gth[oid]:
            gth[oid][x[1]] = []
        gth[oid][x[1]].append([x[2], x[3]])
    return ans,gth


def get_candidate(test_query, rules_dict, learn_edges):
    all_walks = []
    window = 0
    candidates = {}
    score_func = score_12
    args = [[0.1, 0.1]]
    cands_dict = [dict() for _ in range(len(args))]
    dicts_idx = list(range(len(args)))
    cur_ts = test_query[3]
    edges = ra.get_window_edges(data.train_idx, cur_ts, learn_edges, window)

    if test_query[1] in rules_dict:
        for rule in rules_dict[test_query[1]]:
            walk_edges = ra.match_body_relations_complete(rule, edges, test_query[0])
            rule_walks = pd.DataFrame()
            if 0 not in [len(x) for x in walk_edges]:
                rule_walks = ra.get_walks_complete(rule, walk_edges)
                if rule['var_constraints']:
                    rule_walks = ra.check_var_constraints(rule['var_constraints'], rule_walks)
            all_walks.append(rule_walks)
            if not rule_walks.empty:
                # print(rule_walks)
                cands_dict = ra.get_candidates(rule, rule_walks, cur_ts, cands_dict,
                                               score_func, args, dicts_idx)
                top_k_scores = [v for _, v in sorted(cands_dict[0].items(), key=lambda item: item[1],
                                                     reverse=True)][:top_k]
                unique_scores = list(scores for scores, _ in itertools.groupby(top_k_scores))
                if len(unique_scores) >= top_k:
                    break
        if len(cands_dict) > 0:
            candidates = dict(sorted(cands_dict[0].items(), key=lambda item: item[1], reverse=True))
            # print(candidates)
        else:
            candidates = {}
            # print("No candidates found after rule application.")
    else:
        candidates = {}
        # print("No rules exist for this query relation.")
    return candidates

# rule apply when add a new fact

# print(list(neighbors.keys())[:10])

def update(extracted_fact):
    """
    return new facts after the input fact
    :param extracted_fact: new input fact = [s,p,o]
    :param graph: all the graph KB.
    :return: a list of new facts
    """
    test_query_set,gth = get_all_possible_facts(extracted_fact, data)
    # print(test_query_set)

    rules_dict = json.load(open("rules.json"))
    rules_dict = {int(k): v for k, v in rules_dict.items()}
    # print(rules_dict)
    r1 = set([]) # relations only 1-1 mapping, e.g. spouse
    result = []
    for test_query in test_query_set:
        # print("#" * 20)
        # print("test", test_query)

        # print(data.id2entity[test_query[0]], data.id2relation[test_query[1]],
        #       data.id2entity[test_query[2]], data.id2ts[test_query[3]])
        candidates = get_candidate(test_query, rules_dict, learn_edges=graph)
        # print("candidate",test_query,candidates)
        objects = list(candidates.keys())
        if len(objects) > 0:
            for obj in objects:
                if not obj==test_query[0]:
                    if test_query[1] in gth[test_query[0]]:
                        match = False
                        for x in gth[test_query[0]][test_query[1]]:
                            if obj == x[0]:
                                match = True
                                # no change
                                # print("no change")
                                # print(data.id2entity[test_query[0]], data.id2relation[test_query[1]],
                                #       data.id2entity[test_query[2]], data.id2ts[test_query[3]])
                                # f = [data.id2entity[test_query[0]], data.id2relation[test_query[1]],
                                #       data.id2entity[objects[0]], data.id2ts[test_query[3]]]
                                f = [test_query[0],test_query[1],obj]
                                # result.append(f)
                                break
                        if match == False and candidates[obj][0] > 0.01:
                            # update
                            if test_query[1] in r1:
                                # print("update")
                                # print(data.id2entity[test_query[0]], data.id2relation[test_query[1]],
                                #       data.id2entity[objects[0]], data.id2ts[test_query[3]])
                                # f = [data.id2entity[test_query[0]], data.id2relation[test_query[1]],
                                #      data.id2entity[objects[0]], data.id2ts[test_query[3]]]
                                f = [test_query[0],test_query[1],obj]
                                # result.append(f)
                            else:
                                # print("insert")
                                # print(data.id2entity[test_query[0]], data.id2relation[test_query[1]],
                                #       data.id2entity[objects[0]], data.id2ts[test_query[3]])
                                # f = [data.id2entity[test_query[0]], data.id2relation[test_query[1]],
                                #      data.id2entity[objects[0]], data.id2ts[test_query[3]]]
                                f = [test_query[0],test_query[1],obj]
                                result.append(f)

                    elif candidates[obj][0] > 0.01:
                        # insert
                        # print("insert")
                        # print(data.id2entity[test_query[0]], data.id2relation[test_query[1]],
                        #       data.id2entity[objects[0]], data.id2ts[test_query[3]])
                        f = [test_query[0],test_query[1],obj]
                        result.append(f)
    # print(result)
    return result

def load_json(file_path):
    ''' load json file '''
    with open(file_path, "r", encoding="utf8") as f:
        data = json.load(f)
        return data

def fact_eqal(f1,f2):
    return f1[0]==f2[0] and f1[1]==f2[1] and f1[2]==f2[2]

def eval(file_path):
    datas = load_json(file_path)
    total_tp=0
    total_fp=0
    total_fn=0
    for i in range(len(datas)):
        label = [tuple(triple) for triple in datas[i]["subgraph_after"]]
        subgraph_before = [tuple(triple) for triple in datas[i]["subgraph_before"]]
        predict = update(datas[i]["fact"])+datas[i]["subgraph_before"]
        predict = [tuple(triple) for triple in predict]
        # print("label:",label)
        # print("predict:",predict)
        TP = 0
        FP = 0
        FN = 0
        TP = len(set(label) & set(predict))
        # for x in label:
        #     match = False
        #     for y in predict:
        #         if fact_eqal(x,y):
        #             TP+=1
        #             match = True
        #             break
        #     if not match:
        #         FN +=1
        FP = len(set(predict))-TP
        FN = len(set(label))-TP
        total_fp += FP
        total_fn += FN
        total_tp += TP
        # break
       
    epsilon = 1e-30
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall    = total_tp / (total_tp + total_fn + epsilon)
    f1        = 2 * precision * recall / (precision + recall + epsilon)
    print("precision: %.4f,\t recall: %.4f,\t f1: %.4f" % (precision, recall, f1))

if __name__ == '__main__':
    eval(config['news_file'])
    