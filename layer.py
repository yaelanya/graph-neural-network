import numpy as np
from collections import OrderedDict

import functional


class LinerLayer(object):
    def __init__(self, input_dim, output_dim):
        self.params = OrderedDict()
        self.params['weight'] = np.random.normal(0, 0.4, (output_dim, input_dim))
        self.params['bias'] = np.zeros(output_dim)

    def forward(self, x):
        return np.dot(self.params['weight'], x) + self.params['bias']


class GraphLayer(object):
    def __init__(self, dim, T=2):
        self.dim = dim
        self.T = T
        
        self.params = OrderedDict()
        self.params['weight'] = np.random.normal(0, 0.4, (dim, dim))
    
    def forward(self, G):
        self._aggregation(G)
        return self.calc_graph_vec()
                
    def calc_graph_vec(self):
        return self.feats.sum(axis=0)
    
    def _aggregation(self, G):
        #self.feats = np.eye(G.shape[0], self.dim)
        self.feats = np.zeros((G.shape[0], self.dim))
        self.feats[:, 0] = 1.0
            
        for t in range(self.T):
            next_feats = np.zeros(self.feats.shape)
            for i, edges in enumerate(G):
                alpha = self._sum_feat_vec(edges)
                next_feats[i] = functional.ReLU(np.dot(self.params['weight'], alpha))
                
            self.feats = next_feats[:]
            
    def _sum_feat_vec(self, edges):
        edge_indice = np.nonzero(edges)[0]
        return self.feats[edge_indice].sum(axis=0)