"""
this script is used to update files that are needed in knowledge-empowered models
first step: update the csi300_stock_index.npy
second step[for HIST]: update the csi300_market_value_[start_year]to[end_year].npy
third step[for HIST]: update the csi300_stock2concept.npy
fourth step[for RSR, etc.]: update the csi300_multi_stock2stock_[type].npy
"""
import pandas as pd
import numpy as np
import argparse
from neo4j import GraphDatabase
import re
from tqdm import tqdm

class NeoPyExample:

    def __init__(self, url, user, password):
        self.driver = GraphDatabase.driver(url, auth=(user, password))

    def close(self):
        self.driver.close()

    def print_greeting(self, message):
        with self.driver.session() as session:
            greeting = session.write_transaction(self._create_and_return_greeting, message)
            print(greeting)

    def general_query(self, query, parameters=None):
        assert self.driver is not None, "Driver not initialized!"
        session = None
        response = None
        try:
            session = self.driver.session()
            response = list(session.run(query, parameters))
        except Exception as e:
            print("Query failed:", e)
        finally:
            if session is not None:
                session.close()
        return response

    def create_node_query(self, node_type, attributes):
        query = 'CREATE (n:' + node_type + ' $attributes)'
        params = {'attributes': attributes}
        self.general_query(query, parameters=params)

    def create_edge_query(self, relationship, pkn1, pk1, pkn2, pk2):
        query = 'MATCH (a),(b) WHERE a.' + pkn1 + '= $pk1 AND b.' + pkn2 + '= $pk2 CREATE (a)-[r:' + relationship + ']->(b)'
        params = {'pk1': pk1, 'pk2': pk2}
        self.general_query(query, parameters=params)

    def create_write_edge_query(self, scholar_id, paper_id):
        #  (i:institution_entity {cn_name:"{name}"})
        query = 'MATCH (a:scholar_entity {id:$id1}), (b:paper {id:$id2}) CREATE (a) -[r:write]->(b)'
        params = {'id1': scholar_id, 'id2': paper_id}
        self.general_query(query, parameters=params)

    def create_belong_to_industry_edge_query(self, industry_name, industry_id):
        #  (i:institution_entity {cn_name:"{name}"})
        query = 'MATCH (a:company {industry:$id1}), (b:industry {industry_id:$id2}) CREATE (a) -[' \
                'r:belong_to_industry]->(b) '
        params = {'id1': industry_name, 'id2': industry_id}
        self.general_query(query, parameters=params)

    def find_number_of_node(self):
        with self.driver.session() as session:
            result = session.read_transaction(self._find_number_of_node)
            for row in result:
                print("Found node number: {row}".format(row=row))

    def general_read_query(self, query):
        with self.driver.session() as session:
            result = session.read_transaction(self._general_read_query_command, query)
            for row in result:
                print(row[0])
                print('\n')
            return result

    def get_read_query(self, query):
        with self.driver.session() as session:
            result = session.read_transaction(self._general_read_query_command, query)
            res = []
            try:
                for row in result:
                    res.append(row)
            except:
                print("error")
                None
            return res

    def fetch_data_from_neo(self,query):
        with self.driver.session() as session:
            result = session.run(query)
        return result

    def show_type_of_nodes(self):
        query = 'call db.labels()'
        self.general_read_query(query)

    def show_type_of_edges(self):
        query = 'MATCH ()-[relationship]->() RETURN distinct TYPE(relationship) AS type'
        self.general_read_query(query)

    @staticmethod
    def _general_read_query_command(tx, query):
        result = tx.run(query)
        return [row for row in result]

    @staticmethod
    def _create_and_return_greeting(tx, message):
        result = tx.run("CREATE (a:Greeting) "
                        "SET a.message = $message "
                        "RETURN a.message + ', from node ' + id(a)", message=message)
        return result.single()[0]

    @staticmethod
    def _find_number_of_node(tx):
        query = (
            "MATCH (n)"
            "RETURN count(n)"
            "LIMIT 10"
        )
        result = tx.run(query)
        return [row for row in result]


def file_reader(file_name):
    li = []
    for line in open(file_name,'r',encoding='UTF-8'):
        li.append(line)
    return li


def update_index(args):
    csi300 = file_reader(args.calendar_path)
    csi300 = [i.replace('\t','\n') for i in csi300]
    csi300 = [i.split('\n') for i in csi300]
    candidate = []
    for i in csi300:
        if i[2] <= args.index_start_date:
            continue
        elif i[1] >= args.index_end_date:
            continue
        else:
            candidate.append(i[0])

    candidate = list(set(candidate))
    new_stock_index = {}
    for i in range(len(candidate)):
        new_stock_index[candidate[i]] = i
    np.save(args.index_path, new_stock_index)
    return new_stock_index


def update_s2s_matrix(args):
    # default use the index file updated
    stock_index = np.load(args.index_path, allow_pickle=True)
    index_dict = stock_index.tolist()
    all_num_dict = [re.sub("[^0-9]", "", x) for x in index_dict.keys()]
    fintech = NeoPyExample(url=args.neo4j_url, user=args.neo4j_user, password=args.neo4j_pwd)
    tuple_set = []
    for i in tqdm(all_num_dict):
        query = 'match (n:company) -[r1:SW_belongs_to]- (i:SW_industry) -[r2:SW_belongs_to]- (c:company) where n.code contains \'' + i + '\' return n.code, i.name, c.code '
        node = fintech.get_read_query(query)
        node = pd.DataFrame([dict(record) for record in node])
        node.replace('', np.nan, inplace=True)
        node = node.dropna().drop_duplicates()
        node = node.values.tolist()
        tuple_set.extend(node)

    # filter pair that both companies are in csi300
    t_set = [i for i in tuple_set if re.sub("[^0-9]", "", i[2]) in all_num_dict]
    # create industry list
    candidate_industry = list(set([i[1] for i in t_set]))
    # keep digit part of all pairs
    t_set = [[re.sub("[^0-9]", "", i[0]), i[1], re.sub("[^0-9]", "", i[2])] for i in t_set]
    # empty matrix
    adjcent_matrix = np.zeros([len(index_dict), len(index_dict), len(candidate_industry)])
    for i in t_set:
        if i[0][0] == '6':
            sym0 = 'SH' + i[0]
        else:
            sym0 = 'SZ' + i[0]
        if i[2][0] == '6':
            sym1 = 'SH' + i[2]
        else:
            sym1 = 'SZ' + i[2]
        index0 = index_dict[sym0]
        index1 = index_dict[sym1]
        adjcent_matrix[index0][index1][candidate_industry.index(i[1])] = 1
        adjcent_matrix[index0][index0][candidate_industry.index(i[1])] = 1
    for i in range(adjcent_matrix.shape[2]):
        adjcent_matrix[:, :, i] = np.maximum(adjcent_matrix[:, :, i], adjcent_matrix[:, :, i].transpose())
    np.save(args.s2s_static_matrix_path, adjcent_matrix)
    np.save(args.industry_relation_type_path, candidate_industry)

    # dynamic company
    query = 'match (n:company)-[r1]-(c:company) return distinct type(r1)'
    dynamic_node = fintech.get_read_query(query)
    dynamic_node = pd.DataFrame([dict(record) for record in dynamic_node])
    dynamic_node.replace('', np.nan, inplace=True)
    dynamic_node = dynamic_node.dropna().drop_duplicates()
    dynamic_node = dynamic_node.values.tolist()
    # node = [re.findall(r'type=\'(.*?)\'', str(x[0])) for x in node]
    dynamic_node = [x[0] for x in dynamic_node]
    dyset = []
    for i in tqdm(all_num_dict):
        query = 'match (n:company)-[r1]-(c:company) where n.code contains \'' + i + '\' return c.code as code,  r1.time as time, type(r1) '
        node = fintech.get_read_query(query)
        node = pd.DataFrame([dict(record) for record in node])
        node.replace('', np.nan, inplace=True)
        node = node.dropna().drop_duplicates()
        node = node.values.tolist()
        for n in node:
            if re.sub("[^0-9]", "", n[0]) in all_num_dict:
                dyset.append([i, re.sub("[^0-9]", "", n[0]), n[1], n[2]])

    dy_relation_dict = {
        'invest': 0, 'same_industry': 1, 'be_increased_holding': 2, 'cooperate': 3, 'dispute': 4, 'fall': 5,
        'increase_holding': 2, 'reduce_holding': 6, 'rise': 7, 'superior': 8,
        'supply': 9, 'be_invested': 0, 'compete': 10, 'be_reduced_holding': 6, 'be_supplied': 9}

    np.save(args.dynamic_type_list, dy_relation_dict)

    dynamic_adjcent_matrix = np.zeros([len(index_dict), len(index_dict), 11])
    for i in dyset:
        if i[0][0] == '6':
            sym0 = 'SH' + i[0]
        else:
            sym0 = 'SZ' + i[0]
        if i[1][0] == '6':
            sym1 = 'SH' + i[1]
        else:
            sym1 = 'SZ' + i[1]
        index0 = index_dict[sym0]
        index1 = index_dict[sym1]
        dynamic_adjcent_matrix[index0][index1][dy_relation_dict[i[3]]] = 1
        dynamic_adjcent_matrix[index0][index0][dy_relation_dict[i[3]]] = 1

    for i in range(dynamic_adjcent_matrix.shape[2]):
        dynamic_adjcent_matrix[:, :, i] = np.maximum(dynamic_adjcent_matrix[:, :, i],
                                                     dynamic_adjcent_matrix[:, :, i].transpose())

    np.save(args.s2s_dynamic_matrix_path, dynamic_adjcent_matrix)
    hidy = np.concatenate((adjcent_matrix, dynamic_adjcent_matrix), axis=2)
    np.save(args.s2s_hidy_matrix_path, hidy)


def parse_args():
    """
    deliver arguments from json file to program
    :param param_dict: a dict that contains model info
    :return: no return
    """
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--index_start_date', default='2008-01-01')
    parser.add_argument('--index_end_date', default='2023-06-30')
    parser.add_argument('--calendar_path', default='../../stock_model/qlib_data/cn_data/instruments/csi300.txt')
    parser.add_argument('--index_path', default='./data/csi300_stock_index_new.npy')
    parser.add_argument('--s2s_static_matrix_path', default='data/csi300_multi_stock2stock_2023.npy')
    parser.add_argument('--s2s_dynamic_matrix_path', default='data/csi300_multi_stock2stock_dynamic_2023.npy')
    parser.add_argument('--s2s_hidy_matrix_path', default='data/csi300_multi_stock2stock_hidy_2023.npy')

    parser.add_argument('--industry_relation_type_path', default='data/SW_industry_type_list.npy')
    parser.add_argument('--dynamic_type_list', default='data/dynamic_type_list_2023.npy')
    parser.add_argument('--neo4j_url', default='no')
    parser.add_argument('--neo4j_user', default='no')
    parser.add_argument('--neo4j_pwd', default='no')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    input neo4j manually first update index file then the adjcent matrix, we output 3 matrix, one for static industry
    relation, one for dynamic relation which is extracted from news, one for both static and dynamic relation
    """
    args = parse_args()
    update_index(args)
    update_s2s_matrix(args)