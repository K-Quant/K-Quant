from os import remove
import pandas as pd
import math
from tqdm import tqdm
from jaro_winkler import *
import json
import random


# DART Algorithm
class DART:

    def __init__(self, path_dict, srcs_list, domain_list, init_recall_dict, init_sp_dict, alpha=0.5, rou=0.3,
                 default_veracity_score=0.5, iter_num=20, test_mode=False):
        self.path_dict = path_dict
        self.srcs_list = srcs_list
        self.domain_list = domain_list
        self.init_recall_dict = init_recall_dict
        self.init_sp_dict = init_sp_dict
        self.alpha = alpha
        self.rou = rou
        self.default_veracity_score = default_veracity_score
        self.iter_num = iter_num
        self.test_mode = test_mode

    def get_source_domain_richness(self, source_path):
        df = pd.read_csv(source_path).drop(columns=['Unnamed: 0'])
        source_domain_richness = dict()
        for item in self.domain_list:
            source_domain_richness[item] = 0
        for i in range(len(df)):
            key = df["event_type"][i].split('-')[0]
            source_domain_richness[key] += 1
        return source_domain_richness

    def get_global_domain_richness(self, sources_domain_richness):
        domain_richness = dict()
        for item in self.domain_list:
            domain_richness[item] = 0
        for src_key in sources_domain_richness:
            for dm_key in sources_domain_richness[src_key]:
                domain_richness[dm_key] += sources_domain_richness[src_key][dm_key]
        return domain_richness

    def calculate_domain_richness_score(self, sources_global_domain_percent, domain_richness, sources_domain_richness,
                                        alpha=0.5):
        for src_key in sources_domain_richness:
            sources_global_domain_percent[src_key] = dict()
            for dm_key in domain_richness:
                tmp_value = 0
                if domain_richness[dm_key] != 0:
                    tmp_value = sources_domain_richness[src_key][dm_key] / domain_richness[dm_key]
                sources_global_domain_percent[src_key][dm_key] = math.sqrt(1 - math.pow((alpha * tmp_value - 1), 2))
        return sources_global_domain_percent

    def determin_continuity(self, src_df, begin_idx):
        continuity_len = 1
        continuity_list = list()
        begin_id = src_df["id"][begin_idx]
        continuity_list.append(src_df["event_type"][begin_idx].split('-')[0])
        while begin_idx + continuity_len < len(src_df):
            if src_df["id"][begin_idx + continuity_len] == begin_id:
                continuity_len += 1
                continuity_list.append(src_df["event_type"][begin_idx + continuity_len].split('-')[0])
            else:
                return continuity_len, continuity_list
        return continuity_len, continuity_list

    def get_sources_inter_intersection(self):
        sources_inter_intersection = dict()
        df1 = pd.read_csv(self.path_dict["yuncaijing"]).drop(columns=['Unnamed: 0'])
        df2 = pd.read_csv(self.path_dict["10jqka"]).drop(columns=['Unnamed: 0'])
        df3 = pd.read_csv(self.path_dict["eastmoney"]).drop(columns=['Unnamed: 0'])
        sources_inter_intersection["yuncaijing"] = dict()
        sources_inter_intersection["10jqka"] = dict()
        sources_inter_intersection["eastmoney"] = dict()
        for dm_key in self.domain_list:
            sources_inter_intersection["yuncaijing"][dm_key] = list()
            sources_inter_intersection["10jqka"][dm_key] = list()
            sources_inter_intersection["eastmoney"][dm_key] = list()
        for i in range(len(df1)):
            continuity_len, continuity_list = self.determin_continuity(df1, i)
            if continuity_len > 1:
                for dm_key in self.domain_list:
                    if dm_key in continuity_list:
                        for item in continuity_list:
                            if item != dm_key:
                                sources_inter_intersection["yuncaijing"][dm_key].append(item)
        for i in range(len(df2)):
            continuity_len, continuity_list = self.determin_continuity(df2, i)
            if continuity_len > 1:
                for dm_key in self.domain_list:
                    if dm_key in continuity_list:
                        for item in continuity_list:
                            if item != dm_key:
                                sources_inter_intersection["10jqka"][dm_key].append(item)
        for i in range(len(df3)):
            continuity_len, continuity_list = self.determin_continuity(df3, i)
            if continuity_len > 1:
                for dm_key in self.domain_list:
                    if dm_key in continuity_list:
                        for item in continuity_list:
                            if item != dm_key:
                                sources_inter_intersection["eastmoney"][dm_key].append(item)
        return sources_inter_intersection

    def get_source_inter_domain_influence(self, src_inter_intersection, src_domain_richness):
        src_inter_domain_inf = dict()
        for dm_key in self.domain_list:
            src_inter_domain_inf[dm_key] = 0
            if len(src_inter_intersection[dm_key]) != 0:
                non_dm_sum = 0
                for src_dm_key in src_domain_richness:
                    if src_dm_key != dm_key:
                        non_dm_sum += src_domain_richness[src_dm_key]
                if non_dm_sum != 0:
                    src_inter_domain_inf[dm_key] = len(src_inter_intersection[dm_key]) / non_dm_sum
        return src_inter_domain_inf

    # input: src_inter_domain_inf --- inf_s
    # input: src_global_domain_percent --- r_d(s)
    # input: ρ --- [0, 1]
    # When an overlap between different domains is large,
    # ρ should be set higher. When overlap seldom occurs in different domains, ρ should be set lower.
    def get_adjusted_domain_expertise_score(self, rou, src_inter_domain_inf, src_global_domain_percent):
        source_adjusted_domain_expertise = dict()
        for dm_key in self.domain_list:
            tmp_inf_sum = 0
            r_di = src_global_domain_percent[dm_key]
            for inter_key in src_inter_domain_inf:
                if inter_key != dm_key:
                    tmp_inf_sum += r_di * src_inter_domain_inf[inter_key]
            source_adjusted_domain_expertise[dm_key] = r_di + rou * tmp_inf_sum
        return source_adjusted_domain_expertise

    def remove_unknowns(self):
        for idx in range(len(self.triples_df)):
            if self.triples_df["relation"][idx] == "未知Unknown":
                self.triples_df = self.triples_df.drop(idx)
        self.triples_df = self.triples_df.reset_index()
        return

    def extract_undirected_triple(self, index):
        head_code = self.triples_df["head_code"][index]
        tail_code = self.triples_df["tail_code"][index]
        relation = self.triples_df["relation"][index]
        if len(relation) == 3:
            relation = relation[1:]
            temp = head_code
            head_code = tail_code
            tail_code = temp
        return (head_code, tail_code, relation)

    def initialize_rec_and_sp(self, default_recall, default_sp):
        src_domain_recall = dict()
        src_domain_sp = dict()
        for dm_key in self.domain_list:
            src_domain_recall[dm_key] = default_recall
            src_domain_sp[dm_key] = default_sp
        return src_domain_recall, src_domain_sp

    def link_triples2company(self):
        triples_df = self.triples_df
        main_domain_df = pd.read_csv(self.path_dict["main_domain"], header=None)
        for i in range(len(triples_df)):
            # src_name = triples_df["src"][i]
            src_name = self.srcs_list[random.randint(0, 2)]
            # triple = self.extract_undirected_triple(i)
            triple = (triples_df["head_code"][i], triples_df["tail_code"][i], triples_df["relation"][i])
            stock_id = ""
            # if triple[1] in main_domain_df:
            #    stock_id = triple[1]
            # else:
            #    stock_id = triple[0]
            stock_id = triple[0]
            if stock_id not in self.objects_dict:
                self.objects_dict[stock_id] = dict()
                self.objects_dict[stock_id]["datetime"] = list()
                self.objects_dict[stock_id]["triples"] = list()
                self.objects_dict[stock_id]["src"] = list()
                self.objects_dict[stock_id]["triples_label"] = list()
                self.objects_dict[stock_id]["distinct"] = list()
                self.objects_dict[stock_id]["distinct_time"] = list()
                self.objects_dict[stock_id]["distinct_label"] = list()
                self.objects_dict[stock_id]["veracity_value"] = list()
                self.objects_dict[stock_id]["main_domain"] = "财经/交易"
            self.objects_dict[stock_id]["datetime"].append(triples_df["time"][i])
            self.objects_dict[stock_id]["triples"].append(triple)
            self.objects_dict[stock_id]["triples_label"].append(0)
            if stock_id in main_domain_df:
                self.objects_dict[stock_id]["main_domain"] = main_domain_df[stock_id]
            self.objects_dict[stock_id]["src"].append(src_name)
        return

    def link_event2company(self):
        company_df = self.test_df
        company_events = self.objects_dict
        for i in range(len(company_df)):
            stock_id = company_df["stock"][i]
            src_name = company_df["src"][i]
            if stock_id not in company_events:
                company_events[stock_id] = dict()
                company_events[stock_id]["datetime"] = list()
                company_events[stock_id]["event"] = list()
                company_events[stock_id]["event_company"] = list()
                company_events[stock_id]["event_type"] = list()
                company_events[stock_id]["event_label"] = list()
                company_events[stock_id]["src"] = list()
                company_events[stock_id]["distinct"] = list()
                company_events[stock_id]["distinct_type"] = list()
                company_events[stock_id]["distinct_time"] = list()
                company_events[stock_id]["distinct_label"] = list()
                company_events[stock_id]["veracity_value"] = list()
            company_events[stock_id]["datetime"].append(company_df["datetime"][i])
            tmp_event = company_df["text"][i]
            company_events[stock_id]["event"].append(tmp_event)
            company_events[stock_id]["event_company"].append(company_df["argument"][i])
            company_events[stock_id]["event_type"].append(company_df["event_type"][i].split('-')[0])
            company_events[stock_id]["event_label"].append(0)
            company_events[stock_id]["src"].append(src_name)
        return

    def is_same_event(self, text1, text2, time1, time2):
        time1_ymd = time1.split(" ")[0]
        time2_ymd = time2.split(" ")[0]
        if time1_ymd == time2_ymd:
            time1_h = int(time1.split(" ")[1].split(":")[0])
            time2_h = int(time2.split(" ")[1].split(":")[0])
            if abs(time1_h - time2_h) < 2:
                if get_jaro_distance(text1, text2) > 0.8:
                    return True
        return False

    def is_same_company(self, c_alias_name, c_stock):
        company_list = list(self.company_df["argument"])
        if c_alias_name in company_list:
            idx = company_list.index(c_alias_name)
            if c_stock == self.company_df["stock"][idx]:
                return True
        else:
            return False

    def is_same_triple(self, triple1, triple2, time1, time2):
        time1_ymd = time1.split(" ")[0]
        time2_ymd = time2.split(" ")[0]
        if time1_ymd == time2_ymd:
            time1_h = int(time1.split(" ")[1].split(":")[0])
            time2_h = int(time2.split(" ")[1].split(":")[0])
            if abs(time1_h - time2_h) < 2:
                if triple1 == triple2:
                    return True
        return False

    def get_triples_lists(self):
        for stock_id in tqdm(self.objects_dict):
            tag = True
            triples_list = self.objects_dict[stock_id]["triples"]
            for i in range(len(triples_list)):
                current_triple = triples_list[i]
                current_time = self.objects_dict[stock_id]["datetime"][i]
                label = self.srcs_list.index(self.objects_dict[stock_id]["src"][i]) + 1
                for m in range(label - 1):
                    label *= 100
                distinct_list = self.objects_dict[stock_id]["distinct"]
                if tag:
                    tag = False
                    self.objects_dict[stock_id]["triples_label"][i] = 1
                    self.objects_dict[stock_id]["distinct"].append(current_triple)
                    self.objects_dict[stock_id]["distinct_time"].append(current_time)
                    self.objects_dict[stock_id]["distinct_label"].append(label)
                else:
                    tag2 = True
                    for j in range(len(distinct_list)):
                        compared_triple = distinct_list[j]
                        compared_time = self.objects_dict[stock_id]["distinct_time"][j]
                        if current_triple == compared_triple:
                            # print(current_triple, compared_triple, current_time, compared_time)
                            self.objects_dict[stock_id]["distinct_label"][j] += label
                            tag2 = False
                    if tag2:
                        self.objects_dict[stock_id]["triples_label"][i] = 1
                        self.objects_dict[stock_id]["distinct"].append(current_triple)
                        self.objects_dict[stock_id]["distinct_time"].append(current_time)
                        self.objects_dict[stock_id]["distinct_label"].append(label)
        return

    def get_objects_lists(self):
        for stock_id in tqdm(self.objects_dict):
            tag = True
            events_dict = self.objects_dict[stock_id]
            events_list = events_dict["event"]
            for i in range(len(events_list)):
                current_event = events_list[i]
                current_time = events_dict["datetime"][i]
                current_type = events_dict["event_type"][i]
                label = self.srcs_list.index(events_dict["src"][i]) + 1
                for m in range(label - 1):
                    label *= 100
                distinct_list = events_dict["distinct"]
                if len(current_event) > 4 and self.is_same_company(events_dict["event_company"][i], stock_id):
                    if tag:
                        tag = False
                        self.objects_dict[stock_id]["event_label"][i] = 1
                        self.objects_dict[stock_id]["distinct"].append(current_event)
                        self.objects_dict[stock_id]["distinct_time"].append(current_time)
                        self.objects_dict[stock_id]["distinct_label"].append(label)
                        self.objects_dict[stock_id]["distinct_type"].append(current_type)
                    else:
                        tag2 = True
                        for j in range(len(distinct_list)):
                            compared_event = distinct_list[j]
                            compared_time = events_dict["distinct_time"][j]
                            if compared_event == current_event or self.is_same_event(current_event, compared_event,
                                                                                     current_time, compared_time):
                                self.objects_dict[stock_id]["distinct_label"][j] += label
                                tag2 = False
                        if tag2:
                            self.objects_dict[stock_id]["event_label"][i] = 1
                            self.objects_dict[stock_id]["distinct"].append(current_event)
                            self.objects_dict[stock_id]["distinct_time"].append(current_time)
                            self.objects_dict[stock_id]["distinct_label"].append(label)
                            self.objects_dict[stock_id]["distinct_type"].append(current_type)
        return

    def set_veracity_and_confidence(self):
        default_veracity_score = self.default_veracity_score
        for object_key in self.objects_dict.keys():
            self.claimed_value[object_key] = dict()
            for src_name in self.srcs_list:
                self.objects_dict[object_key][src_name] = list()
            # Initialize veracity score
            for value in self.objects_dict[object_key]["distinct"]:
                self.objects_dict[object_key]["veracity_value"].append(default_veracity_score)

            # obtain the number of the object in each source 
            each_value_of_src = list()
            each_value_of_src.append(0)
            each_value_of_src.append(0)
            each_value_of_src.append(0)
            for i in range(len(self.objects_dict[object_key]["src"])):
                src_name = self.objects_dict[object_key]["src"][i]
                src_idx = self.srcs_list.index(src_name)
                each_value_of_src[src_idx] += 1

            if len(each_value_of_src) < len(self.srcs_list):
                num = len(self.srcs_list) - len(each_value_of_src)
                for i in range(num):
                    each_value_of_src.append(0)

            # calculate the confidence score: c_s(o)
            Vo = len(self.objects_dict[object_key]["distinct"])
            for idx in range(len(each_value_of_src)):
                src_key = self.srcs_list[idx]
                V_so = each_value_of_src[idx]
                VominusV_so = Vo - V_so
                self.claimed_value[object_key][src_key] = dict()
                if V_so != 0 and Vo != 0:
                    self.claimed_value[object_key][src_key]["claimed"] = (1 - VominusV_so / (Vo * Vo)) * (1 / V_so)
                    self.claimed_value[object_key][src_key]["unclaimed"] = 1 / (Vo * Vo)
                else:
                    self.claimed_value[object_key][src_key]["claimed"] = 0
                    self.claimed_value[object_key][src_key]["unclaimed"] = 0
                for label in self.objects_dict[object_key]["distinct_label"]:
                    if label >= 30000:
                        if label % 10000 != 0 or label % 1000 != 0:
                            if idx == 0:
                                self.objects_dict[object_key][src_key].append("claimed")
                            if idx == 2:
                                self.objects_dict[object_key][src_key].append("claimed")
                            if idx == 1:
                                if label % 100 != 0 or label % 10 != 0:
                                    self.objects_dict[object_key][src_key].append("claimed")
                                else:
                                    self.objects_dict[object_key][src_key].append("unclaimed")
                        else:
                            if idx == 0:
                                self.objects_dict[object_key][src_key].append("unclaimed")
                            if idx == 2:
                                self.objects_dict[object_key][src_key].append("claimed")
                            if idx == 1:
                                if label % 100 != 0 or label % 10 != 0:
                                    self.objects_dict[object_key][src_key].append("claimed")
                                else:
                                    self.objects_dict[object_key][src_key].append("unclaimed")
                    if label >= 200 and label < 30000:
                        if label % 100 != 0 or label % 10 != 0:
                            if idx != 2:
                                self.objects_dict[object_key][src_key].append("claimed")
                            else:
                                self.objects_dict[object_key][src_key].append("unclaimed")
                        else:
                            if idx == 0:
                                self.objects_dict[object_key][src_key].append("unclaimed")
                            if idx == 1:
                                self.objects_dict[object_key][src_key].append("claimed")
                            if idx == 2:
                                self.objects_dict[object_key][src_key].append("unclaimed")
                    if label < 200:
                        if idx == 0:
                            self.objects_dict[object_key][src_key].append("claimed")
                        if idx == 1:
                            self.objects_dict[object_key][src_key].append("unclaimed")
                        if idx == 2:
                            self.objects_dict[object_key][src_key].append("unclaimed")
        return

    def get_main_domain(self):
        domain_dict_value = dict()
        for object_key in self.objects_dict.keys():
            for item in self.domain_list:
                domain_dict_value[item] = 0
            stock_events = self.objects_dict[object_key]
            for domain in stock_events["event_type"]:
                domain_dict_value[domain] += 1
            main_domain = self.domain_list[0]
            value = 0
            for domain in domain_dict_value:
                if domain_dict_value[domain] > value:
                    main_domain = domain
                    value = domain_dict_value[domain]
            stock_events["main_domain"] = main_domain
        return

    def update_veracity_score(self):
        for object_key in self.objects_dict.keys():
            main_domain = self.objects_dict[object_key]["main_domain"]
            for src_idx in range(len(self.srcs_list)):
                src_name = self.srcs_list[src_idx]
                claimed_confidence = self.claimed_value[object_key][src_name]["claimed"]
                unclaimed_confidence = self.claimed_value[object_key][src_name]["unclaimed"]
                expertise_score_src_domain = self.sources_adjusted_domain_expertise[src_name][main_domain]
                prob_value_true = 1
                prob_value_false = 1
                for value_idx in range(len(self.objects_dict[object_key][src_name])):
                    # domain = self.objects_dict[object_key]["distinct_type"][value_idx]
                    expertise_score_src_domain = self.sources_adjusted_domain_expertise[src_name][main_domain]
                    value = self.objects_dict[object_key][src_name][value_idx]
                    if value == "claimed":
                        factor_power = claimed_confidence * expertise_score_src_domain
                        prob_value_true = prob_value_true * math.pow(self.sources_domain_recall[src_name][main_domain],
                                                                     factor_power)
                        prob_value_false = prob_value_false * math.pow(
                            1 - self.sources_domain_recall[src_name][main_domain], factor_power)
                    if value == "unclaimed":
                        factor_power = unclaimed_confidence * expertise_score_src_domain
                        prob_value_true = prob_value_true * math.pow(1 - self.sources_domain_sp[src_name][main_domain],
                                                                     factor_power)
                        prob_value_false = prob_value_false * math.pow(self.sources_domain_sp[src_name][main_domain],
                                                                       factor_power)

                    # update veracity score
                    previous_veracity = self.objects_dict[object_key]["veracity_value"][value_idx]
                    if prob_value_true != 0 and previous_veracity != 0:
                        inter_value = ((1 - previous_veracity) / previous_veracity) * (
                                    prob_value_false / prob_value_true)
                        self.objects_dict[object_key]["veracity_value"][value_idx] = 1 / (1 + inter_value)
                    else:
                        self.objects_dict[object_key]["veracity_value"][value_idx] = 0
        return

    def get_source_object_dict(self):
        source_object_dict = dict()
        for src_name in self.srcs_list:
            source_object_dict[src_name] = dict()
            for dom in self.domain_list:
                source_object_dict[src_name][dom] = list()
                for obj in self.objects_dict:
                    if self.objects_dict[obj]["main_domain"] == dom:
                        source_object_dict[src_name][dom].append(obj)
        return source_object_dict

    def update_source_recall(self):
        for src_name in self.srcs_list:
            for dom in self.domain_list:
                veracity_sum = 0
                claim_value_count = 0
                for obj in self.source_object_dict[src_name][dom]:
                    for value in self.objects_dict[obj][src_name]:
                        if value == "claimed":
                            value_idx = self.objects_dict[obj][src_name].index(value)
                            veracity_sum += self.objects_dict[obj]["veracity_value"][value_idx]
                            claim_value_count += 1
                    if claim_value_count != 0:
                        self.sources_domain_recall[src_name][dom] = veracity_sum / claim_value_count
        return

    def update_source_specificity(self):
        for src_name in self.srcs_list:
            for dom in self.domain_list:
                veracity_sum = 0
                unclaim_value_count = 0
                for obj in self.source_object_dict[src_name][dom]:
                    for value in self.objects_dict[obj][src_name]:
                        if value == "unclaimed":
                            value_idx = self.objects_dict[obj][src_name].index(value)
                            veracity_sum += 1 - self.objects_dict[obj]["veracity_value"][value_idx]
                            unclaim_value_count += 1
                    if unclaim_value_count != 0:
                        self.sources_domain_sp[src_name][dom] = veracity_sum / unclaim_value_count
        return

    def truth_inference(self, theta):
        infer_truth_dict = dict()
        for obj in self.objects_dict:
            if obj not in infer_truth_dict.keys():
                infer_truth_dict[obj] = dict()
                infer_truth_dict[obj]["events"] = list()
                infer_truth_dict[obj]["datetime"] = list()
                infer_truth_dict[obj]["veracity_value"] = list()
            for idx in range(len(self.objects_dict[obj]["veracity_value"])):
                veracity_value = self.objects_dict[obj]["veracity_value"][idx]
                if veracity_value > theta:
                    infer_truth_dict[obj]["events"].append(self.objects_dict[obj]["distinct"][idx])
                    infer_truth_dict[obj]["datetime"].append(self.objects_dict[obj]["distinct_time"][idx])
                    infer_truth_dict[obj]["veracity_value"].append(veracity_value)
        return infer_truth_dict

    def truth_inference_triples(self, theta):
        infer_truth_dict = dict()
        for obj in self.objects_dict:
            if obj not in infer_truth_dict.keys():
                infer_truth_dict[obj] = dict()
                infer_truth_dict[obj]["triples"] = list()
                infer_truth_dict[obj]["datetime"] = list()
                infer_truth_dict[obj]["veracity_value"] = list()
            for idx in range(len(self.objects_dict[obj]["veracity_value"])):
                veracity_value = self.objects_dict[obj]["veracity_value"][idx]
                if veracity_value > theta:
                    infer_truth_dict[obj]["triples"].append(self.objects_dict[obj]["distinct"][idx])
                    infer_truth_dict[obj]["datetime"].append(self.objects_dict[obj]["distinct_time"][idx])
                    infer_truth_dict[obj]["veracity_value"].append(veracity_value)
                else:
                    print(obj, self.objects_dict[obj]["distinct"][idx])
        return infer_truth_dict

    def integrate_bayesian(self, theta=0.5):
        count_iter = 0
        while count_iter < self.iter_num:
            self.update_veracity_score()
            self.update_source_recall()
            self.update_source_specificity()
            count_iter += 1
        if self.test_mode:
            infer_truth_dict = self.truth_inference(theta)
        else:
            infer_truth_dict = self.truth_inference_triples(theta)
        return infer_truth_dict

    def evaluation(self, pred_list, previous_true_num=0):
        gt_df = self.test_df
        total_num = len(self.test_df)
        acc = list()
        true_num = 0
        for i in range(len(gt_df)):
            if pred_list[i] == int(gt_df["label"][i]):
                true_num += 1
        final_acc = true_num / total_num
        return final_acc

    def load_company_alias_dict(self, path):
        with open(path, "r") as f:
            company_dict = json.load(f)
        company_df = []
        for k, v in company_dict.items():
            if v["stock"] == [] or v["stock"][0] == "":
                continue
            for alias in v["alias"]:
                company_df.append([k, alias, v["stock"][0]])
        company_df = pd.DataFrame(company_df, columns=["company_name", "argument", "stock"])

        return company_df

    def run(self):

        # Firstly, estimate the quality confidence of data source by domain richness
        sources_global_domain_percent = dict()  # calculated each domain percentage for each data source
        sources_domain_richness = dict()  # record domain quantity number for each data source
        for src in self.srcs_list:
            sources_domain_richness[src] = self.get_source_domain_richness(self.path_dict[src])
        domain_richness = self.get_global_domain_richness(sources_domain_richness)
        sources_global_domain_percent = self.calculate_domain_richness_score(sources_global_domain_percent,
                                                                             domain_richness, sources_domain_richness,
                                                                             self.alpha)
        sources_inter_intersection = self.get_sources_inter_intersection()

        sources_inter_domain_influence = dict()
        for src in self.srcs_list:
            sources_inter_domain_influence[src] = self.get_source_inter_domain_influence(
                sources_inter_intersection[src], sources_domain_richness[src])

        self.sources_adjusted_domain_expertise = dict()
        for src in self.srcs_list:
            self.sources_adjusted_domain_expertise[src] = self.get_adjusted_domain_expertise_score(self.rou,
                                                                                                   sources_inter_domain_influence[
                                                                                                       src],
                                                                                                   sources_global_domain_percent[
                                                                                                       src])

        self.sources_domain_recall = dict()
        self.sources_domain_sp = dict()
        for src in self.srcs_list:
            self.sources_domain_recall[src], self.sources_domain_sp[src] = self.initialize_rec_and_sp(
                self.init_recall_dict[src], self.init_sp_dict[src])

        # Secondly, estimate multi-truth confidence
        if self.test_mode:
            # self.company_df = self.load_company_alias_dict(self.path_dict["company"])
            self.test_df = pd.read_csv(self.path_dict["testset"])
            self.objects_dict = dict()
            self.triples_df = pd.read_csv(self.path_dict["triples"])
            # self.remove_unknowns()
            self.link_triples2company()
            self.get_triples_lists()

        else:
            self.objects_dict = dict()
            self.triples_df = pd.read_csv(self.path_dict["triples"])
            self.remove_unknowns()
            self.link_triples2company()
            self.get_triples_lists()

        self.claimed_value = dict()
        self.set_veracity_and_confidence()
        # self.get_main_domain()

        # Thirdly, integrate Bayesian model
        self.source_object_dict = self.get_source_object_dict()

        self.infer_truth_dict = self.integrate_bayesian()

        # If it is in the test mode, run the following.
        if self.test_mode:
            label_list = list()
            event_list = list()
            for object_key in self.objects_dict.keys():
                stock_dict = self.objects_dict[object_key]
                for idx in range(len(stock_dict["triples_label"])):
                    label_list.append(stock_dict["triples_label"][idx])
            print("Accuracy of DART is: ", self.evaluation(label_list))

        print(self.sources_domain_recall, self.sources_domain_sp)

        return


# Init DART recall and sp
def prior_source_recall(rawData):
    sources_recall_dict = dict()
    sources_recall_dict["yuncaijing"] = 0
    sources_recall_dict["10jqka"] = 0
    sources_recall_dict["eastmoney"] = 0
    total_true_num = 0
    for idx in range(len(rawData)):
        label = rawData["Correct"][idx]
        if label:
            total_true_num += 1
            if rawData["S1"][idx] == 1:
                sources_recall_dict["yuncaijing"] += 1
            if rawData["S2"][idx] == 1:
                sources_recall_dict["10jqka"] += 1
            if rawData["S3"][idx] == 1:
                sources_recall_dict["eastmoney"] += 1
    sources_recall_dict["yuncaijing"] = sources_recall_dict["yuncaijing"] / total_true_num
    sources_recall_dict["10jqka"] = sources_recall_dict["10jqka"] / total_true_num
    sources_recall_dict["eastmoney"] = sources_recall_dict["eastmoney"] / total_true_num
    return sources_recall_dict


def prior_source_specificity(rawData):
    sources_specificity_dict = dict()
    sources_specificity_dict["yuncaijing"] = 0
    sources_specificity_dict["10jqka"] = 0
    sources_specificity_dict["eastmoney"] = 0
    total_false_num = 0
    for idx in range(len(rawData)):
        label = rawData["Correct"][idx]
        if not label:
            total_false_num += 1
            if rawData["S1"][idx] == 0:
                sources_specificity_dict["yuncaijing"] += 1
            if rawData["S2"][idx] == 0:
                sources_specificity_dict["10jqka"] += 1
            if rawData["S3"][idx] == 0:
                sources_specificity_dict["eastmoney"] += 1
    sources_specificity_dict["yuncaijing"] = sources_specificity_dict["yuncaijing"] / total_false_num
    sources_specificity_dict["10jqka"] = sources_specificity_dict["10jqka"] / total_false_num
    sources_specificity_dict["eastmoney"] = sources_specificity_dict["eastmoney"] / total_false_num
    return sources_specificity_dict


def init_recall_sp(rawData_df):
    sources_recall_dict = prior_source_recall(rawData_df)
    sources_specificity_dict = prior_source_specificity(rawData_df)
    return sources_recall_dict, sources_specificity_dict