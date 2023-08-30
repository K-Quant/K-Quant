class GraphExplainer:
    '''
    For 'rnn+gnn+predictor' model structure, this is an abstract interpreter to generate graph explanations.
    '''

    def __init__(self, graph_model, num_layers, device):
        self.graph_model = graph_model
        self.num_layers = num_layers
        self.device = device

    def explain(self, full_model, graph, stkid):
        return None

    def explaination_to_graph(self, explanation, subgraph, stkid): # form the explanations to a graph
        return subgraph, stkid
