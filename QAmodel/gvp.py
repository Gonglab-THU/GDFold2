import torch
import functools
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from utils import *

class GVP(nn.Module):
    '''
    From https://github.com/drorlab/gvp-pytorch

    Geometric Vector Perceptron.
    in_dims: tuple (n_scalar, n_vector)
    out_dims: tuple (n_scalar, n_vector)
    h_dim: intermediate number of vector channels, optional
    activations: tuple of functions (scalar_act, vector_act)
    vector_gate: whether to use vector gating. If 'True', vector_act will be used 
        as sigma^+ in vector gating.
    '''
    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi: 
            self.h_dim = h_dim or max(self.vi, self.vo) 
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)
        
        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))
        
    def forward(self, x):
        '''
        x: tuple (s, V) of 'torch.Tensor', or a single 
            'torch.Tensor' (if vectors_in is 0)
        return: tuple (s, V) of 'torch.Tensor', or a single
            'torch.Tensor' (if vectors_out is 0)
        '''
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)    
            vn = norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo: 
                v = self.wv(vh) 
                v = torch.transpose(v, -1, -2)
                if self.vector_gate: 
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(
                        norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                                device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)
        
        return (s, v) if self.vo else s


class LayerNorm(nn.Module):
    '''
    From https://github.com/drorlab/gvp-pytorch

    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)
        
    def forward(self, x):
        '''
        x: tuple (s, V) of 'torch.Tensor', or single 'torch.Tensor' 
        '''
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn


class _VDropout(nn.Module):
    '''
    From https://github.com/drorlab/gvp-pytorch

    Vector channel dropout where the elements of each
    vector channel are dropped together.
    '''
    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        x: 'torch.Tensor' corresponding to vector channels
        '''
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli((1 - self.drop_rate) * torch.ones(x.shape[:-1], 
                                device=device)).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x
    

class Dropout(nn.Module):
    '''
    From https://github.com/drorlab/gvp-pytorch

    Combined dropout for tuples (s, V). Takes tuples (s, V) 
        as input and as output.
    '''
    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        '''
        x: tuple (s, V) of 'torch.Tensor', or single 'torch.Tensor' 
            (will be assumed to be scalar channels)
        '''
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)
    
    
class GVPConv(MessagePassing):
    '''
    From https://github.com/drorlab/gvp-pytorch

    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings, and 
        returns new node embeddings.
    
    in_dims: input node embedding dimensions (n_scalar, n_vector)
    out_dims: output node embedding dimensions (n_scalar, n_vector)
    edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    n_layers: number of GVPs in the message function
    module_list: preconstructed message function, overrides n_layers
    aggr: should be "add" if some incoming edges are masked, as in
        a masked autoregressive decoder architecture, otherwise "mean"
    activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    vector_gate: whether to use vector gating. If 'True', vector_act will be used 
        as sigma^+ in vector gating.
    '''
    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean",
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVPConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        
        GVP_ = functools.partial(GVP, activations=activations, 
                                 vector_gate=vector_gate)
        
        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), 
                        (self.so, self.vo), activations=(None, None)))
            else:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), out_dims))
                
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims,
                                       activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        '''
        x: tuple (s, V) of 'torch.Tensor'
        edge_index: array of shape [2, n_edges]
        edge_attr: tuple (s, V) of 'torch.Tensor'
        '''
        x_s, x_v = x
        message = self.propagate(edge_index, s=x_s, 
                                 v=x_v.reshape(x_v.shape[0], 3*x_v.shape[1]),
                                 edge_attr=edge_attr)
        return split(message, self.vo) 

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1]//3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1]//3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return merge(*message)
    

class GVPConvLayer(nn.Module):
    '''
    From https://github.com/drorlab/gvp-pytorch

    Full graph convolution / message passing layer with 
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward 
    network to node embeddings, and returns updated node embeddings.
    
    node_dims: node embedding dimensions (n_scalar, n_vector)
    edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    n_message: number of GVPs to use in message function
    n_feedforward: number of GVPs to use in feedforward function
    drop_rate: drop probability in all dropout layers
    autoregressive: if 'True', this 'GVPConvLayer' will be used with a 
        different set of input node embeddings for messages where src >= dst
    activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    vector_gate: whether to use vector gating. If 'True', vector_act will be used 
        as sigma^+ in vector gating.
    '''
    def __init__(self, node_dims, edge_dims, n_message=3, 
                 n_feedforward=2, drop_rate=.1, autoregressive=False, 
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVPConvLayer, self).__init__()
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                           aggr="add" if autoregressive else "mean",
                           activations=activations, vector_gate=vector_gate)
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4*node_dims[0], 2*node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward-2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

    def forward(self, x, edge_index, edge_attr,
                autoregressive_x=None, node_mask=None):
        '''
        x: tuple (s, V) of 'torch.Tensor'
        edge_index: array of shape [2, n_edges]
        edge_attr: tuple (s, V) of 'torch.Tensor'
        autoregressive_x: tuple (s, V) of 'torch.Tensor'. If not 'None', 
            will be used as src node embeddings for forming messages 
            where src >= dst. The corrent node embeddings 'x' will still be 
            the base of the update and the pointwise feedforward.
        node_mask: array of type 'bool' to index into the first dim of node 
            embeddings (s, V). If not 'None', only these nodes will be updated.
        '''
        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)
            
            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward))
            
            count = scatter_add(torch.ones_like(dst), dst,
                                dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)
            
            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)
        else:
            dh = self.conv(x, edge_index, edge_attr)
        
        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)
            
        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))
        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))
        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x
    