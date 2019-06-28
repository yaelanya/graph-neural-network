from collections import OrderedDict

import layer
import functional


class GNNModel(object):
    def __init__(self, dim):
        self.layers = OrderedDict()
        
        self.graph_layer = layer.GraphLayer(dim)
        self.layers['l_graph'] = self.graph_layer
        
        self.liner_layer = layer.LinerLayer(dim, 1)
        self.layers['l_liner'] = self.liner_layer
        
        
    def forward(self, G):
        x = self.graph_layer.forward(G)
        x = self.liner_layer.forward(x)
        
        return x
    
    def predict(self, G):
        x = self.forward(G)
        pred_proba = functional.sigmoid(x)
        pred = pred_proba > 0.5
        
        return pred