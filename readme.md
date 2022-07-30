# Pytorch MONet implementation

[![Python build package and test](https://github.com/Michedev/MONet-pytorch/actions/workflows/build-and-test.yaml/badge.svg)](https://github.com/Michedev/MONet-pytorch/actions/workflows/build-and-test.yaml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/monet-pytorch)
![PyPI - Downloads](https://img.shields.io/pypi/dm/monet-pytorch)
![PyPI - Status](https://img.shields.io/pypi/status/monet-pytorch)


Pytorch implementation of [Multi-Object Network(MONet)](https://arxiv.org/abs/1901.11390)

![monet_architecture](https://user-images.githubusercontent.com/12683228/179543891-11392837-a5a1-4f8d-b601-72525f208fe0.png)


# How to install

You can install through pip with the following command

    pip install monet-pytorch

or clone this repository locally and install with [poetry](https://python-poetry.org/)

    git clone https://github.com/Michedev/MONet-pytorch
    cd MONet-pytorch
    poetry install
## How to use

The package comes with a set of predefined configurations based on paper specifications, namely _monet_ and _monet-iodine_ (MONet as defined in IODINE paper).

    from monet_pytorch import Monet
    
    monet = Monet.from_config(model='monet')

There is also another custom architecture _monet-lightweight_ which has less parameters than the original ones.

Furthermore, the model architecture slightly changes based on the dataset (e.g. U-Net blocks) 
picked between the ones cited in MONet paper (_CLEVR 6, Multidsprites colored on colored, 
Multidsprited colored on grayscale, Tetrominoes_). 

    from monet_pytorch import Monet
    
    monet = Monet.from_config(model='monet', dataset='tetrominoes')


In alternative, you can set custom dataset parameters through the function arguments

    from monet_pytorch import Monet
    
    monet = Monet.from_config(model='monet', dataset_width=48, dataset_height=48, scene_max_objects=6)

Lastly, you can make your custom MONet by input your custom configuration as OmegaConf DictConfig

    from monet_pytorch import Monet

    custom_monet_config = OmegaConf.create("""
    dataset:
      width: 44
      height: 44
      max_num_objects: 10
    model:  #this config file follows MONet implementation from IODINE paper
      _target_: monet_pytorch.model.Monet
      height: ${dataset.height}
      width: ${dataset.width}
      num_slots: ${dataset.max_num_objects}
      name: monet-iodine
      bg_sigma: 0.32
      fg_sigma: 0.1
      beta_kl: 0.43
      gamma: 0.5
      latent_size: 16
      input_channels: 3
      encoder:
        _target_: torch.nn.Sequential
        _args_:
          - _target_: monet_pytorch.template.sequential_cnn.make_sequential_cnn_from_config
            channels: [44, 44, 32, 14]
            kernels: 3
            strides: 2
            paddings: 0
            input_channels: 4
            batchnorms: true
            bn_affines: false
            activations: relu
          - _target_: torch.nn.Flatten
            start_dim: 1
          - _target_: torch.nn.Linear
            in_features: 256
            out_features: 256
          - _target_: torch.nn.ReLU
          - _target_: torch.nn.Linear
            in_features: 256
            out_features: ${prod:${model.latent_size},2}
      decoder:
        _target_: monet_pytorch.template.encoder_decoder.BroadcastDecoderNet
        w_broadcast: ${sum:${dataset.width},8}
        h_broadcast: ${sum:${dataset.height},8}
        net:
          _target_: monet_pytorch.template.sequential_cnn.make_sequential_cnn_from_config
          input_channels: ${sum:${model.latent_size},2} # latent size + 2
          channels: [32, 32, 64, 64, 4]  # last is 4 channels because rgb (3) + mask (1)
          kernels: [3, 3, 3, 3, 1]
          paddings: 0
          activations: [relu, relu, relu, relu, null]  #null means no activation function no activation
          batchnorms: [true, true, true, true, false]
          bn_affines: [false, false, false, false, false]
      unet:
        _target_: monet_pytorch.unet.UNet
        input_channels: ${model.input_channels}
        num_blocks: 5
        filter_start: 16
        mlp_size: 128""")

    custom_monet: Monet = Monet.from_custom_config(custom_monet_config)

# Model performances

This implementation reproduce very closely ARI MONet's values

## Special thanks

I would like to thank @apra and @addtt for the help to fix code bugs and to improve model performances
