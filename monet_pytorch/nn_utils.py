from collections import OrderedDict
from typing import List, Union, Literal, Optional

import torch
from torch import nn

ActivationFunctionEnum = Union[Literal['relu', 'leakyrelu', 'elu', 'glu'], None]


@torch.no_grad()
def init_trunc_normal(model, mean=0, std: float = 1):
    for name, tensor in model.named_parameters():
        if 'bias' in name:
            tensor.zero_()
        elif 'weight' in name:
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)


@torch.no_grad()
def init_xavier_(model):
    # print("Xavier init")
    for name, tensor in model.named_parameters():
        if name.endswith('.bias'):
            tensor.zero_()
        elif len(tensor.shape) == 1:
            pass  # silent
            # print(f"    skipped tensor '{name}' because shape="
            #       f"{tuple(tensor.shape)} and it does not end with '.bias'")
        else:
            torch.nn.init.xavier_uniform_(tensor)


def norm_gradient(model, p_norm):
    return sum(param.grad.norm(p_norm) for param in model.parameters())


def get_activation_module(act_f_name, try_inplace=True):
    if act_f_name == 'leakyrelu':
        ActF = torch.nn.LeakyReLU()
    elif act_f_name == 'elu':
        ActF = torch.nn.ELU()
    elif act_f_name == 'relu':
        ActF = torch.nn.ReLU(inplace=try_inplace)
    elif act_f_name == 'glu':
        ActF = torch.nn.GLU(dim=1)  # channel dimension in images
    elif act_f_name == 'sigmoid':
        ActF = torch.nn.Sigmoid()
    elif act_f_name == 'tanh':
        ActF = torch.nn.Tanh()
    else:
        raise ValueError(f"act_f_name = {act_f_name} not found")
    return ActF


def calc_output_shape_conv(width: int, height: int, kernels: List[int], paddings: List[int], strides: List[int]):
    for kernel, stride, padding in zip(kernels, strides, paddings):
        width = (width + 2 * padding - kernel) // stride + 1
        height = (height + 2 * padding - kernel) // stride + 1
    return width, height


def print_num_params(model: nn.Module, max_depth: Optional[int] = 4):
    """Prints overview of the model with number of parameters.

    Optionally, it groups together parameters below a certain depth in the
    module tree.

    Args:
        model (torch.nn.Module)
        max_depth (int, optional)
    """

    sep = '.'  # string separator in parameter name
    print("\n--- Trainable parameters:")
    num_params_tot = 0
    num_params_dict = OrderedDict()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        num_params = param.numel()

        if max_depth is not None:
            split = name.split(sep)
            prefix = sep.join(split[:max_depth])
        else:
            prefix = name
        if prefix not in num_params_dict:
            num_params_dict[prefix] = 0
        num_params_dict[prefix] += num_params
        num_params_tot += num_params
    for n, n_par in num_params_dict.items():
        print("{:8d}  {}".format(n_par, n))
    print("  - Total trainable parameters:", num_params_tot)
    print("---------\n")


def get_debug_information_model(model):
    hooks = []

    def printnorm(self, input, output):
        # input is a tuple of packed inputs
        # output is a Tensor. output.data is the Tensor we are interested
        print(self.__class__.__name__ + ' forward')
        print('')
        print('input: ', type(input))
        print('input[0]: ', type(input[0]))
        print('output: ', type(output))
        print('')
        print('input size:', input[0].size())
        if isinstance(output, dict):
            for name, value in output.items():
                print('output key: {} | size: {}'.format(name, value.data.size()))
                print('output key: {} |  norm:{}'.format(name, value.data.norm()))
        else:
            print('output size: {}'.format(output.data.size()))
            print('output norm:{}'.format(output.data.norm()))

        print('')

    def simplified_printnorm(self, input, output):
        # input is a tuple of packed inputs
        # output is a Tensor. output.data is the Tensor we are interested

        if isinstance(output, dict):
            print('{:>25} | {:>25} | {:>25}'.format(self.__class__.__name__, str(list(input[0].size())),
                                                    "Dictionary Output"))
            for name, value in output.items():
                print('{:>25} | {:>25} | {:>25} | {}'.format(" ", " ", str(list(value.data.size())), name))
                print('{:>25} | {:>25} | {:>25.2e} | {}'.format(" ", " ", value.data.norm().item(), name))

        else:
            print('{:>25} | {:>25} | {:>25}'.format(self.__class__.__name__, str(list(input[0].size())),
                                                    str(list(output.data.size()))))
            print('{:>25} | {:>25.2e} | {:>25.2e}'.format(" ", input[0].norm().item(), output.data.norm().item()))

        print('')

    for module in model.modules():
        hooks.append(module.register_forward_hook(simplified_printnorm))

    return hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()
