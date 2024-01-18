import torch
import torch.nn as nn
from typing import Iterable, Optional
import random

'''
Inspired from https://github.com/kytimmylai/NoisyNN-PyTorch/tree/main, but I wrap it as a function to inject noise into any layer's output given its name.
There should be two way to implement the random layer choosing:
    - At each step, randomly choose a layer, register forward hook for that layer and remove it afther the forward pass. (Might cause performance issue? Need to check)
    - Register hook for each layer, then performing a random choosing operator at each forward pass. (This should considerably harder but might overcone the )

'''

class Chosen:
    '''
    just hold the value of the chosen layer name
    '''
    def  __init__(self, n_layers: int = 1) -> None:
        self.name = None
        self.n_layers = n_layers

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
            print(f'Layer {name} is chosen!')
            # shape = output.shape
            # output = output.reshape(shape[0],-1)
            # k = torch.prod(torch.tensor(shape[1:]))
            # linear_noise = optimal_quality_matrix(k).to(output.device)
            # output = output@linear_noise
            # output = output.view(*shape)

        return output
    return hook

def random_chooser(handles: dict, chosen: Chosen):
    def hook(model, input):
        keys = list(handles.keys())
        keys.remove('choser')
        chosen.name = random.choice(keys)

    return hook


def inject_noisy_nn(model: nn.Module, layers_name: Iterable[str] = ['stages.3.blocks.2','stages.2.blocks.26'], n_layers_inject_per_batch: int = 1, inplace = True, verbose = True):
    if getattr(model, "is_noisy", False):
        if verbose:
            print(f"The model is already populated with NoisyNN instances. Please consider removing them before adding inject new NoisyNN instance.")
        return model
    handles = {}
    chosen = Chosen(n_layers_inject_per_batch)
    layers_name = list(*layers_name)

    if not inplace:
        from copy import deepcopy
        model = deepcopy(model)
    max_idx = None
    for idx, (name, child) in enumerate(model.named_modules()):
        if name in layers_name:
            handles[name] = (child.register_forward_hook(add_noise(name= name, chosen= chosen))) 
            layers_name.remove(name)
            if verbose:
                print(f"NoisyNN instance injected onto layer {name}")

    if verbose:
        if len(layers_name) > 0:
            repr = f'Incompatible layer(s): {layers_name}'
        else:
            repr = "<--All chosen layers are injected with NoisyNN instance!-->"

        print(repr)

    handles["choser"] = model.register_forward_pre_hook(random_chooser(handles, chosen))
    setattr(model, "noisy_handles", handles)
    setattr(model, "is_noisy", True)
    return model    

def remove_noisy_nn(model: nn.Module, inplace = True, verbose = True):
    if not getattr(model, "is_noisy", False):
        if verbose:
            print("No NoisyNN instance left in the model!")
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

    print("<--All injected NoisyNN instances have been removed!-->")

    return model


