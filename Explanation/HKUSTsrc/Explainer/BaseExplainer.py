from abc import ABC, abstractmethod


class BaseExplainer(ABC):
    def __init__(self, model_to_explain):
        self.model_to_explain = model_to_explain


    @abstractmethod
    def prepare(self, args):
        """Prepars the explanation method for explaining.
        Can for example be used to train the method"""
        pass

    @abstractmethod
    def explain(self, feature, graph):
        """
        Main method for explaining samples
        :param feature: index of node/graph in self.graphs
        :return: explanation for sample
        """
        pass

