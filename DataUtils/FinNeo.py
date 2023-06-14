from neo4j import GraphDatabase
import pandas as pd
import numpy as np
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


class FinNeo(NeoPyExample):
    def __init__(self, **kwargs):
        '''
        url:数据库接口
        user:user name
        password:user password
        '''
        super().__init__(kwargs['url'], kwargs['user'], kwargs['pwd'])
