import torch
import torch.nn as nn
from typing import Iterable, Optional
import random
import math
'''
https://github.com/kytimmylai/NoisyNN-PyTorch/tree/main implementation seem wrong as it only does linear transformation noise on 1 dimension(W instead of CxHxW)
There should be two way to implement the random layer choosing:
    - At each step, randomly choose a layer, register forward hook for that layer and remove it afther the forward pass. (Might cause performance issue? Need to check)
    - Register hook for each layer, then performing a random choosing operator at each forward pass. (This should considerably harder but might overcone the )

'''

class Chosen:
    '''
    just hold the value of the chosen layer name
    '''
    def  __init__(self) -> None:
        self.name = None

def quality_matrix(k, alpha=0.3):
    """r
    Quality matrix Q. Described in the eq (17) so that eps = QX, where X is the input. 
    Alpha is 0.3, as mentioned in Appendix D.
    """
    identity = torch.diag(torch.ones(k))
    shift_identity = torch.zeros(k, k)
    for i in range(k):
        shift_identity[(i+1)%k, i] = 1
    opt = -alpha * identity + alpha * shift_identity
    return opt

def optimal_quality_matrix(k):
    """r
    Optimal Quality matrix Q. Described in the eq (19) so that eps = QX, where X is the input. 
    Suppose 1_(kxk) is torch.ones
    """
    return torch.diag(torch.ones(k)) * -k/(k+1) + torch.ones(k, k) / (k+1)


def add_noise(name, chosen: Chosen):
    def hook(model: nn.Module, input, output: torch.Tensor):
        if name == chosen.name and model.training:
            # print(f'Layer {name} is chosen!')
            shape = output.shape
            old_shape = None
            if len(shape) == 3:
                old_shape = shape
                output = output.permute(0,2,1).view(shape[0],shape[2], math.sqrt(shape[1]), math.sqrt(shape[1]))
                shape = output.shape
            k = shape[-1]
            linear_noise = optimal_quality_matrix(k).to(output.device)
            output = output@linear_noise + output
            output = output.view(*shape)
            if old_shape is not None:
                output = output.view(shape[0], shape[1], -1).permute(0,2,1)
        return output
    return hook

def random_chooser(handles: dict, chosen: Chosen):
    def hook(model, input):
        keys = list(handles.keys())
        keys.remove('choser')
        chosen.name = random.choice(keys)

    return hook


def inject_noisy_nn(model: nn.Module, layers_name: Iterable[str] = ['stages.3.blocks.2','stages.2.blocks.26'], inplace = True):
    if getattr(model, "is_noisy", False):
        return model
    handles = {}
    chosen = Chosen()

    if not inplace:
        from copy import deepcopy
        model = deepcopy(model)
    max_idx = None
    for idx, (name, child) in enumerate(model.named_modules()):
        if name in layers_name:
            handles[name] = (child.register_forward_hook(add_noise(name= name, chosen= chosen))) 

    handles["choser"] = model.register_forward_pre_hook(random_chooser(handles, chosen))
    setattr(model, "noisy_handles", handles)
    # model.register_forward_hook(hack_output(output_dict= intermediate_outputs))
    setattr(model, "is_noisy", True)
    return model    

def remove_noisy_nn(model: nn.Module, inplace = True, verbose = True):
    if not getattr(model, "is_noisy", False):
        print("No NoisyNN layer left in the model!")
        return model
    
    print("Trying to remove NoisyNN!!!")

    if not inplace:
        from copy import deepcopy
        model = deepcopy(model)

    for name, handle in model.noisy_handles.items():
        handle.remove()
        if verbose:
            print(f"NoisyNN instance removed from layer {name}.")

    delattr(model, 'is_noisy')
    delattr(model, 'noisy_handles')

    print("Done removing NoisyNN!!!")

    return model


