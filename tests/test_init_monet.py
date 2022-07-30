import torch.nn
from omegaconf import OmegaConf, DictConfig

from monet_pytorch import Monet


def test_init_plain_monet():
    model = Monet.from_config()
    assert model.bg_sigma == 0.09
    assert model.width == 64
    assert model.height == 64
    assert model.num_slots == 5
    assert model.decoder.w_broadcast == (64 + 8)
    assert model.decoder.h_broadcast == (64 + 8)


def test_tetrominoes_monet():
    model = Monet.from_config(dataset='tetrominoes')
    assert model.bg_sigma == 0.09
    assert model.width == 32
    assert model.height == 32
    assert model.num_slots == 4
    assert model.decoder.w_broadcast == (32 + 8)
    assert model.decoder.h_broadcast == (32 + 8)


def test_tetrominoes_monet_iodine():
    model = Monet.from_config(model='monet-iodine', dataset='tetrominoes')
    assert model.bg_sigma == 0.06
    assert model.width == 32
    assert model.height == 32
    assert model.num_slots == 4
    assert model.decoder.w_broadcast == (32 + 8)
    assert model.decoder.h_broadcast == (32 + 8)


def test_monet_lightweight():
    model = Monet.from_config(model='monet-lightweight')
    assert model.width == 64
    assert model.num_slots == 5
    assert model.bg_sigma == 0.1


def test_tetrominoes_monet_lightweight():
    model = Monet.from_config(model='monet-lightweight', dataset='tetrominoes')
    assert model.bg_sigma == 0.23
    assert model.width == 32
    assert model.height == 32
    assert model.num_slots == 4
    assert model.decoder.w_broadcast == (32 + 4)
    assert model.decoder.h_broadcast == (32 + 4)


def test_custom_config():
    custom_monet_config: DictConfig = OmegaConf.create("""
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
    assert custom_monet.bg_sigma == 0.32
    exp_out_channels = [44, 44, 32, 14]
    for layer in custom_monet.encoder.modules():
        if isinstance(layer, torch.nn.Conv2d):
            exp_out_channel = exp_out_channels.pop(0)
            assert layer.out_channels == exp_out_channel
    assert custom_monet.bg_sigma == 0.32
    assert custom_monet.fg_sigma == 0.1
    assert custom_monet.unet.filter_start == 16
    assert custom_monet.beta_kl == 0.43
