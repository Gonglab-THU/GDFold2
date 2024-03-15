import torch

def tuple_cat(*args, dim=-1):
    '''
    From https://github.com/drorlab/gvp-pytorch

    Concatenates any number of tuples (s, V) elementwise.
    dim: dimension along which to concatenate when viewed
        as the 'dim' index for the scalar-channel tensors.
    '''
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


def tuple_sum(*args):
    '''
    From https://github.com/drorlab/gvp-pytorch

    Sums any number of tuples (s, V) elementwise.
    '''
    return tuple(map(sum, zip(*args)))


def merge(s, v):
    '''
    From https://github.com/drorlab/gvp-pytorch

    Merges a tuple (s, V) into a single 'torch.Tensor', where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use '_split(x, nv)' to reverse.
    '''
    v = torch.reshape(v, v.shape[:-2] + (3*v.shape[-2],))
    return torch.cat([s, v], -1)


def tuple_index(x, idx):
    '''
    From https://github.com/drorlab/gvp-pytorch

    Indexes into a tuple (s, V) along the first dimension.
    idx: any object which can be used to index into a 'torch.Tensor'
    '''
    return x[0][idx], x[1][idx]


def split(x, nv):
    '''
    From https://github.com/drorlab/gvp-pytorch

    Splits a merged representation of (s, V) back into a tuple. 
    Should be used only with '_merge(s, V)' and only if the tuple 
        representation cannot be used.
    x: the 'torch.Tensor' returned from '_merge'
    nv: the number of vector channels in the input to '_merge'
    '''
    v = torch.reshape(x[..., -3*nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3*nv]
    return s, v


def norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    From https://github.com/drorlab/gvp-pytorch
    
    L2 norm of tensor clamped above a minimum value 'eps'.
    sqrt: if 'False', returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out