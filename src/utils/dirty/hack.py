import torch
import torch.nn as nn
from typing import Iterable, Optional

def rec_getattr(__o, __name: str, /, __default = None):
    if hasattr(__o, __name):
        return getattr(__o, __name)
    else:
        __attr, __next_pos_attrs = __name.split(".", 1)
        try:
            return rec_getattr(getattr(__o, __attr), __next_pos_attrs, __default)
        except:
            return None
    
def rec_setattr(__o, __name: str, __value):
    if hasattr(__o, __name):
        return setattr(__o, __name, __value)
    else:
        __attr, __next_pos_attrs = __name.split(".", 1)
        return rec_setattr(getattr(__o, __attr), __next_pos_attrs, __value)
    

def rec_add_module(__o, __name, __module):
    if '.' in __name:
        __pre, __post = __name.split(".", 1)
        rec_add_module(getattr(__o, __pre), __post, __module)
    else:
        __o.add_module(__name, __module)

def get_pre_output(name, output_dict: dict):
    def hook(model, input):
        output_dict[name] = input
    return hook
def get_output(name, output_dict: dict):
    def hook(model, input, output):
        output_dict[name] = output
    return hook

def hack_features_output(output_dict: dict, hack_forward = True):
    def hook(model, input, output):
        # output_dict["feature_out"] = output
        if hack_forward:
            return output_dict
    return hook


def hack_output(output_dict: dict):
    def hook(model, input, output):
        output_dict["classifier_out"] = output
        return output_dict
    return hook


def inject_output_hack(model: nn.Module, layers_name: Optional[Iterable[str]] = ['stages.3.blocks.2','stages.2.blocks.26'], inplace = True) -> nn.Module:
    if getattr(model, "is_hacked", False):
        return model
    intermediate_outputs = {}
    removed_layers = {}
    output_hack_handles = {}
    if not inplace:
        from copy import deepcopy
        model = deepcopy(model)
    max_idx = None
    for idx, (name, child) in enumerate(model.named_modules()):
        if name in layers_name:
            output_hack_handles[name] = child.register_forward_hook(get_output(name= name, output_dict= intermediate_outputs))
            max_idx = idx

    for idx, (name, child) in enumerate(model.named_modules()):
        if idx > max_idx:
            removed_layers[name] = child
            rec_add_module(model, name, nn.Identity())

    output_hack_handles['output_layer'] = model.register_forward_hook(hack_features_output(output_dict= intermediate_outputs))
    # model.register_forward_hook(hack_output(output_dict= intermediate_outputs))
    setattr(model, "is_hacked", True)
    setattr(model, "output_hack_handles", output_hack_handles)
    setattr(model, 'removed_layers', removed_layers)
    return model

def remove_output_hack(model: nn.Module, inplace = True, verbose = True):
    if not getattr(model, "is_hacked", False):
        print("No output hack found in model!")
        return model
    
    print("Trying to remove output hack!!!")

    if not inplace:
        from copy import deepcopy
        model = deepcopy(model)

    for name, handle in model.output_hack_handles.items():
        handle.remove()
        if verbose:
            print(f"Output caching removed from layer {name}.")

    for name, child in model.removed_layers.items():
        rec_add_module(model, name, child)
        if verbose:
            print(f"Layer {name} restored to the original.")

    delattr(model, "is_hacked")
    delattr(model, "output_hack_handles")

    print("Done removing output hack!!!")

    return model
    
