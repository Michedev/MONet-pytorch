# Pytorch MONet implementation

[![Python build package and test](https://github.com/Michedev/MONet-pytorch/actions/workflows/build-and-test.yaml/badge.svg)](https://github.com/Michedev/MONet-pytorch/actions/workflows/build-and-test.yaml)

Pytorch implementation of [Multi-Object Network(MONet)](https://arxiv.org/abs/1901.11390)

# How to install

You can install through pip with the following command

    pip install monet-pytorch

or clone this repository locally and install with [poetry](https://python-poetry.org/)

    git clone https://github.com/Michedev/MONet-pytorch
    cd MONet-pytorch
    poetry install
## How to use

The package comes with a set of predefined configurations based on paper specifications, namely _monet_ and _monet-iodine_ (MONet as defined in IODINE paper).

    from monet_pytorch import init_monet
    
    monet = init_monet(model='monet')

There is also another custom architecture _monet-lightweight_ which has less parameters than the original ones.

Furthermore, the model architecture slightly changes based on the dataset (e.g. U-Net blocks) 
picked between the ones cited in MONet paper (_CLEVR 6, Multidsprites colored on colored, 
Multidsprited colored on grayscale, Tetrominoes_). 

    from monet_pytorch import init_monet
    
    monet = init_monet(model='monet', dataset='tetrominoes')


In alternative, you can set custom dataset parameters through the function arguments

    from monet_pytorch import init_monet
    
    monet = init_monet(model='monet', dataset_width=48, dataset_height=48, scene_max_objects=6)

Lastly, you can make your custom MONet by input your custom configuration as OmegaConf DictConfig

    from monet_pytorch import init_monet_custom    

    custom_monet = OmegaConf.create("""
    dataset:
      width: 48
      height: 48
      max_num_objects: 4
    model:
      _target_: monet_pytorch.model.Monet
      height: 48
      width: 48
      num_slots: 5
      name: monet-custom
      bg_sigma: 0.32
      fg_sigma: 0.35
      num_blocks_unet: 4
      beta_kl: 0.5
      gamma: 0.5
      latent_size: 16
      channels_unet: 16
      encoder_config:
        channels: [44, 44, 32, 14]
        kernels: 3
        strides: 2
        paddings: 0
        input_channels: 4
        batchnorms: true
        bn_affines: false
        activations: relu
        mlp_hidden_size: 256
        mlp_output_size: 32  # latent_size * 2
      decoder_config:
        w_broadcast: ${add:${dataset.width},8}
        h_broadcast: ${add:${dataset.height},8}
        input_channels: 18 # latent size + 2
        channels: [32, 32, 64, 64, 4]  # last is 4 channels because rgb (3) + mask (1)
        kernels: [3, 3, 3, 3, 1]
        paddings: 0
        activations: [relu, relu, relu, relu, null]  #null means no activation function no activation
        batchnorms: [true, true, true, true, false]
        bn_affines: [false, false, false, false, false]
    """)

    custom_monet = init_monet_custom(custom_monet)

