import torch
from torch import nn
import numpy as np

def encode_position(input, levels, inc_input):
    """
    For each scalar, we encode it using a series of sin() and cos() functions with different frequency.
        - With L pairs of sin/cos function, each scalar is encoded to a vector that has 2L elements. Concatenating with
          itself results in 2L+1 elements.
        - With C channels, we get C(2L+1) channels output.

    :param input:   (..., C)            torch.float32
    :param levels:  scalar L            int
    :return:        (..., C*(2L+1))     torch.float32
    """

    # this is already doing 'log_sampling' in the official code.
    result_list = [input] if inc_input else []
    temp = torch.cat([2.**i * input for i in range(levels)],-1)
    result_list.append(torch.sin(temp))
    result_list.append(torch.cos(temp))

    result_list = torch.cat(result_list, dim=-1)  # (..., C*(2L+1)) The list has (2L+1) elements, with (..., C) shape each.
    return result_list  # (..., C*(2L+1))

def barf_encode_position(input, levels, inc_input, progress=1):
    result_list = [input] if inc_input else []
    progress = progress*2+0.1
    res = encode_position(input, levels, False) # 
    k = torch.arange(levels, dtype=torch.float32, device=input.device)
    weight = (1-(progress * levels - k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
    shape = res.shape
    result_list.append((res.view(-1,levels) * weight).view(*shape))
    return torch.cat(result_list, -1)

class hash_position(nn.Module):
    def __init__(self,bounding_box, n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        pass

    def __call__(input):
        pass
    
