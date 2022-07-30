from typing import Union, List, Tuple

import torch.nn
from omegaconf import ListConfig
from torch import nn as nn

from monet_pytorch.nn_utils import get_activation_module, ActivationFunctionEnum

def _scalars_to_list(params: dict):
    param_names = list(params.keys())
    list_size = len(params['channels'])
    for k in param_names:
        if k not in ['input_channels', 'channels']: # not resizable parameters
            if not isinstance(params[k], (tuple, list, ListConfig)):
                params[k] = [params[k]] * list_size
    return params


def make_sequential_cnn_from_config(input_channels: int, channels: List[int], kernels: Union[List[int], int],
                                    batchnorms: Union[List[bool], bool] = False, bn_affines: Union[List[bool], bool] = False,
                                    paddings: Union[List[int], int] = 0, strides: Union[List[int], int] = 1,
                                    activations: Union[List[ActivationFunctionEnum], ActivationFunctionEnum] = 'relu',
                                    output_paddings: Union[List[int], int] = 0, conv_transposes: Union[List[bool], bool] = False,
                                    return_params: bool = False, try_inplace_activation: bool = True,
                                    layernorms: Union[List[bool], bool] = False,
                                    ln_affines: Union[List[bool], bool] = False) -> Union[torch.nn.Sequential, Tuple[torch.nn.Sequential, dict]]:
    """
    Generate a sequential CNN based on input parameters. All the parameters, except for channels can be either a list
    of values or a scalar (in such case the value will be spread to all the layers.)
    Furthermore, this Sequential generation procedure is very suitable for yaml-based configuration.
    Args:
        input_channels ():
        channels (): the output channels of each layer
        kernels (): kernel size of each conv layer
        batchnorms (): set true if every conv layer is followed by a batch norm layer
        bn_affines (): set true if every batch norm layer should have learnable parameters
        paddings (): padding of each conv layer
        strides (): stride of each conv layer
        activations (): activation function after the conv or batch norm layer
        output_paddings (): output padding of each conv transpose layer. Set only if conv_trasposes[i] is true
        conv_transposes (): set true to use Conv2dTranspose instead of Conv2d
        return_params (): return also a dictionary containing all the configuration parameters
        try_inplace_activation (): set true to use inplace activation with ReLU
        layernorms (): set true to use layer norm after the conv layer.
                       Note that batchnorms[i] and layernorms[i] cannot be both true
        ln_affines (): set true to have learnable parameters in layernorm

    Returns: the Sequential module and optionally the input parameters as dictionary

    """
    params = locals()
    params = _scalars_to_list(params)
    for i in range(len(params['channels'])):
        assert not (params['batchnorms'][i] and params['layernorms'][i]), \
               f'batchnorms {params["batchnorms"]}  layernorms {params["layernorms"]} ' \
               f'cannot be both true at the same index'
    for k in list(params):
        if k != 'input_channels':
            assert len(params[k]) == len(params['channels']), f'len({k}) = {len(params[k])} ' \
                                                              f'!= len(channels) = {len(params["channels"])}\n' \
                                                              f'{k} = {params[k]} - channels = {params["channels"]}'
    layers = []
    input_channels = input_channels
    layer_infos = zip(params['channels'], params['batchnorms'],
                      params['bn_affines'], params['kernels'],
                      params['strides'], params['paddings'],
                      params['activations'], params['conv_transposes'],
                      params['output_paddings'], params['layernorms'],
                      params['ln_affines'])
    it_counter = 0
    for i in range(len(params['channels'])):
        channel = params['channels'][i]
        if params['conv_transposes'][i]:
            layers.append(nn.ConvTranspose2d(input_channels, channel, params['kernels'][i],
                                             params['strides'][i], params['paddings'][i], params['output_paddings'][i]))
        else:
            layers.append(nn.Conv2d(input_channels, channel, params['kernels'][i],
                                    params['strides'][i], params['paddings'][i]))

        if params['batchnorms'][i]:
            layers.append(nn.BatchNorm2d(channel, affine=params['bn_affines'][i]))
        if params['layernorms'][i]:
            layers.append(nn.GroupNorm(1, channel, affine=params['ln_affines'][i]))
        if params['activations'][i] is not None:
            layers.append(get_activation_module(params['activations'][i], try_inplace=try_inplace_activation))
        input_channels = channel
        if params['activations'][i] == 'glu':
            input_channels = input_channels // 2
        it_counter += 1
    assert it_counter == len(channels), f"Number of iteration {it_counter} is different from number of layers {len(channels)}, " \
                                       f"try to check if any config parameter is shorted than channels length"
    if not return_params:
        return nn.Sequential(*layers)
    else:
        return nn.Sequential(*layers), params