from omegaconf import OmegaConf

from monet_pytorch import init_monet, init_monet_custom


def test_init_plain_monet():
    model = init_monet()
    assert model.bg_sigma == 0.09
    assert model.encoder_config.channels == [32, 32, 64, 64]
    assert model.decoder_config.channels == [32, 32, 64, 64, 4]
    assert model.width == 64
    assert model.height == 64
    assert model.num_slots == 5
    assert model.decoder_config.w_broadcast == (64 + 8)
    assert model.decoder_config.h_broadcast == (64 + 8)


def test_tetrominoes_monet():
    model = init_monet(dataset='tetrominoes')
    assert model.bg_sigma == 0.09
    assert model.width == 32
    assert model.height == 32
    assert model.num_slots == 4
    assert model.decoder_config.w_broadcast == (32 + 8)
    assert model.decoder_config.h_broadcast == (32 + 8)


def test_tetrominoes_monet_iodine():
    model = init_monet(model='monet-iodine', dataset='tetrominoes')
    assert model.bg_sigma == 0.06
    assert model.width == 32
    assert model.height == 32
    assert model.num_slots == 4
    assert model.decoder_config.w_broadcast == (32 + 8)
    assert model.decoder_config.h_broadcast == (32 + 8)


def test_tetrominoes_monet_lightweight():
    model = init_monet(model='monet-lightweight', dataset='tetrominoes')
    assert model.bg_sigma == 0.23
    assert model.width == 32
    assert model.height == 32
    assert model.num_slots == 4
    assert model.decoder_config.w_broadcast == (32 + 4)
    assert model.decoder_config.h_broadcast == (32 + 4)
    assert model.encoder_config.channels == [16, 16, 32, 32]
    assert model.decoder_config.channels == [16, 16, 4]

def test_custom_config():
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
    assert custom_monet.encoder_config.channels == [44, 44, 32, 14]
    assert custom_monet.bg_sigma == 0.32
    assert custom_monet.fg_sigma == 0.35
    assert custom_monet.channels_unet == 16