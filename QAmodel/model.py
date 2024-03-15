import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from gvp import GVP, GVPConvLayer, LayerNorm

class MQAModel(nn.Module):
    """
    GVP-GNN for Model Quality Assessment. From https://github.com/drorlab/gvp-pytorch

    node_in_dim: node dimensions in input graph.
    node_h_dim: node dimensions to use in GVP-GNN layers
    edge_in_dim: edge dimensions in input graph
    edge_h_dim: edge dimensions to embed to before use in GVP-GNN layers
    seq_in: if 'True', sequences will also be passed in with the forward 
        pass; otherwise, sequence information is assumed to be part of 
        input node embeddings
    num_layers: number of GVP-GNN layers
    drop_rate: rate to use in all dropout layers
    """
    def __init__(self, node_in_dim, node_h_dim, edge_in_dim, edge_h_dim, 
                 seq_in=False, num_layers=8, drop_rate=0.1):
        super(MQAModel, self).__init__()
        if seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim = (node_in_dim[0] + 20, node_in_dim[1])
        
        self.W_v = nn.Sequential(LayerNorm(node_in_dim),
                                 GVP(node_in_dim, node_h_dim, activations=(None, None)))
        
        self.W_e = nn.Sequential(LayerNorm(edge_in_dim),
                                 GVP(edge_in_dim, edge_h_dim, activations=(None, None)))
        
        self.layers = nn.ModuleList(GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
                                    for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(LayerNorm(node_h_dim), 
                                   GVP(node_h_dim, (ns, 0)))
            
        self.dense = nn.Sequential(nn.Linear(ns, 2*ns), 
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=drop_rate),
                                   nn.Linear(2*ns, 1))
        
    def forward(self, h_V, edge_index, h_E, seq=None, batch=None):      
        """
        h_V: tuple (s, V) of node embeddings
        edge_index: 'torch.Tensor' of shape [2, num_edges]
        h_E: tuple (s, V) of edge embeddings
        seq: if not 'None', int 'torch.Tensor' of shape [num_nodes] 
            to be embedded and appended to 'h_V'
        """
        if seq is not None:
            seq = self.W_s(seq)
            h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])
        
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)
        
        if batch is None: 
            out = out.mean(dim=0, keepdims=True)
        else: 
            out = scatter_mean(out, batch, dim=0)
        
        out = self.dense(out).squeeze(-1)
        return torch.sigmoid(out)