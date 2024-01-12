import json

from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from tqdm import tqdm
# fintech = FinNeo(url="bolt://143.89.126.57:5001",user='neo4j',pwd='csproject')

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
            result = session.execute_read(self._general_read_query_command, query)
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


class FinNeo(NeoPyExample):
    def __init__(self, **kwargs):
        '''
        url:数据库接口
        user:user name
        password:user password
        '''
        super().__init__(kwargs['url'], kwargs['user'], kwargs['pwd'])


if __name__ == '__main__':
    import re
    stock_index = np.load('D:\ProjectCodes\K-Quant\Data\csi300_stock_index.npy', allow_pickle=True)
    index_dict = stock_index.tolist()
    all_num_dict = [re.sub("[^0-9]", "", x) for x in index_dict.keys()]
    fintech = FinNeo(url="bolt://143.89.126.57:5001", user='neo4j', pwd='csproject')

    tuple_set = []
    query = 'match (i:industry) return i.name as name'
    industry_list = pd.DataFrame([dict(record) for record in fintech.get_read_query(query)])
    industry_list = industry_list.values.tolist()
    industry_list = [x[0] for x in industry_list]
    print(industry_list)
    for industry in tqdm(industry_list):
        temp_set = []
        for i in all_num_dict:
            query = 'match (n:company) -[r1]- (i:industry) -[r2]- (c:company) where n.code contains \'' + i + '\' and i.name = \'' + industry + '\' return c.code as code '
            node = fintech.get_read_query(query)
            node = pd.DataFrame([dict(record) for record in node])
            node.replace('', np.nan, inplace=True)
            node = node.dropna().drop_duplicates()
            node = node.values.tolist()
            node = [x[0] for x in node]
            for n in node:
                if re.sub("[^0-9]", "", n) in all_num_dict:
                    temp_set.append([i, re.sub("[^0-9]", "", n)])
        tuple_set.append(temp_set)

    t_set = []
    valid_industry = []
    for s in range(len(tuple_set)):
        if len(tuple_set[s]) != 0:
            t_set.append(tuple_set[s])
            valid_industry.append(industry_list[s])

    print(valid_industry)

    s2s_list = []
    for s2s_set in t_set:
        static_neighbor = np.zeros([len(index_dict), len(index_dict)])
        for group in s2s_set:
            if group[0][0] == '6':
                sym0 = 'SH' + group[0]
            else:
                sym0 = 'SZ' + group[0]
            if group[1][0] == '6':
                sym1 = 'SH' + group[1]
            else:
                sym1 = 'SZ' + group[1]
            index0 = index_dict[sym0]
            index1 = index_dict[sym1]
            static_neighbor[index0][index1] = 1
        s2s_list.append(static_neighbor)

    yh_neighbor = np.transpose(np.array(s2s_list))
    # 4561个二元组，4561*2=9122
    print(yh_neighbor.shape)
    print(np.sum(yh_neighbor))

    # save first_neightbor as csi300_stock2stock
    np.save(r'D:\ProjectCodes\K-Quant\Data\new_graph_data\csi300_multi_stock2stock.npy', yh_neighbor)

    compete = np.expand_dims(np.load(r'D:\ProjectCodes\K-Quant\Data\Raw_relationship\compete.npy'), axis=2)
    cooprate = np.expand_dims(np.load(r'D:\ProjectCodes\K-Quant\Data\Raw_relationship\cooprate.npy'), axis=2)
    dispute = np.expand_dims(np.load(r'D:\ProjectCodes\K-Quant\Data\Raw_relationship\dispute.npy'), axis=2)
    fall = np.expand_dims(np.load(r'D:\ProjectCodes\K-Quant\Data\Raw_relationship\fall.npy'), axis=2)
    increase_holding = np.expand_dims(np.load(r'D:\ProjectCodes\K-Quant\Data\Raw_relationship\increase_holding.npy'), axis=2)
    invest = np.expand_dims(np.load(r'D:\ProjectCodes\K-Quant\Data\Raw_relationship\invest.npy'), axis=2)
    reduce_holding = np.expand_dims(np.load(r'D:\ProjectCodes\K-Quant\Data\Raw_relationship\reduce_holding.npy'), axis=2)
    rise = np.expand_dims(np.load(r'D:\ProjectCodes\K-Quant\Data\Raw_relationship\rise.npy'), axis=2)
    same_industry = np.expand_dims(np.load(r'D:\ProjectCodes\K-Quant\Data\Raw_relationship\same_industry.npy'), axis=2)
    superior = np.expand_dims(np.load(r'D:\ProjectCodes\K-Quant\Data\Raw_relationship\superior.npy'), axis=2)
    supply = np.expand_dims(np.load(r'D:\ProjectCodes\K-Quant\Data\Raw_relationship\supply.npy'), axis=2)
    dy_list = [compete, cooprate, dispute, fall, increase_holding, invest, reduce_holding, rise, same_industry,
               superior,
               supply]
    dy_relation = np.concatenate(dy_list, axis=2)
    print(dy_relation.shape)

    sw1 = np.expand_dims(np.load(r'D:\ProjectCodes\K-Quant\Data\Raw_relationship\csi300_stock2stock_SWL1.npy'), axis=2)
    sw2 = np.expand_dims(np.load(r'D:\ProjectCodes\K-Quant\Data\Raw_relationship\csi300_stock2stock_SWL2.npy'), axis=2)
    sw3 = np.expand_dims(np.load(r'D:\ProjectCodes\K-Quant\Data\Raw_relationship\csi300_stock2stock_SWL3.npy'), axis=2)
    sw_realtion = np.concatenate([sw1, sw2, sw3], axis=2)

    hold = np.expand_dims(np.load(r'D:\ProjectCodes\K-Quant\Data\Raw_relationship\csi300_stock2stock_hold.npy'), axis=2)
    hold2 = np.expand_dims(np.load(r'D:\ProjectCodes\K-Quant\Data\Raw_relationship\csi300_stock2stock_hold_2_hop.npy'), axis=2)
    hold_relation = np.concatenate([hold, hold2], axis=2)
    print(hold_relation.shape)

    all_relation = np.concatenate([yh_neighbor, dy_relation, sw_realtion, hold_relation], axis=2)
    print(all_relation.shape)
    np.save(r'D:\ProjectCodes\K-Quant\Data\new_graph_data\csi300_multi_stock2stock.npy', yh_neighbor)
    np.save(r'D:\ProjectCodes\K-Quant\Data\new_graph_data\csi300_multi_stock2stock_all.npy', all_relation)
    dy_relation_names = ["compete", "cooprate", "dispute", "fall", "increase_holding", "invest", "reduce_holding", "rise", "same_industry",
     "superior","supply"]

    sw_relation_names = ["sw1", "sw2", "sw3"]

    hold_names = ["hold", "hold_2_hop"]

    names = valid_industry + dy_relation_names + sw_relation_names + hold_names

    print(names)
    json_file_path = r"D:\ProjectCodes\K-Quant\Data\new_graph_data\name_list.json"
    with open(json_file_path, 'w') as json_file:
        json.dump(names, json_file)





