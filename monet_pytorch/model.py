from math import prod
from typing import Union, Literal, Optional

import hydra
import torch
import torch.distributions as dists
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn import functional as F

from monet_pytorch.attention_net import AttentionNet
from monet_pytorch.template.encoder_decoder import BroadcastDecoderNet
from monet_pytorch.paths import CONFIG_MODEL, CONFIG_DATASET, CONFIG_SPECIAL_CASES
from monet_pytorch.unet import UNet

_MODEL_CONFIG_VALUES = ['monet', 'monet-iodine', 'monet-lightweight']
_DATASET_CONFIG_VALUES = ['clevr_6', 'multidsprites_colored_on_grayscale',
                          'tetrominoes', 'multidsprites_colored_on_colored']


class Monet(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        latent_size: int,
        num_slots: int,
        beta_kl: float,
        gamma: float,
        encoder: Union[torch.nn.Module, torch.nn.Sequential],
        decoder: Union[torch.nn.Module, torch.nn.Sequential, BroadcastDecoderNet],
        unet: Union[torch.nn.Module, UNet],
        input_channels: int = 3,
        bg_sigma: float = 0.09,
        fg_sigma: float = 0.11,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        name: str = 'monet'
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.latent_size = latent_size
        self.num_slots = num_slots
        self.beta_kl = beta_kl
        self.gamma = gamma
        self.encoder = encoder
        self.decoder = decoder
        self.unet = unet
        self.input_channels = input_channels
        self.bg_sigma = bg_sigma
        self.fg_sigma = fg_sigma
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.name = name

        self.attention_net = AttentionNet(self.unet)
        self.beta_orig = self.beta_kl
        self.prior_dist = dists.Normal(self.prior_mean, self.prior_std)
        self.register_buffer('lower_bound_mask', torch.FloatTensor([1e-5]))  # for backward compatibility

    def forward(self, x):
        scope_shape = list(x.shape)
        scope_shape[1] = 1
        log_scope = torch.zeros(scope_shape, device=x.device)
        log_masks = []
        for i in range(self.num_slots - 1):
            log_mask, log_scope = self.attention_net(x, log_scope)
            log_masks.append(log_mask)
        log_masks.append(log_scope)
        log_masks = torch.cat(log_masks, dim=1)

        slots_shape = list(x.shape)
        slots_shape.insert(1, self.num_slots)
        slots = torch.zeros(slots_shape, device=x.device)

        kl_zs, masks_pred, neg_log_p_xs, zs = self.forward_vae_slots(log_masks, slots, x)
        log_masks_pred = torch.cat(masks_pred, dim=1).log_softmax(dim=1)
        masks = log_masks.exp()
        neg_log_p_xs = torch.cat(neg_log_p_xs, dim=1)
        masks_pred = log_masks_pred.exp()

        kl_masks, loss, neg_log_p_xs = self._calc_loss(kl_zs, masks, masks_pred, neg_log_p_xs)
        return {'loss': loss,
             'neg_log_p_x': neg_log_p_xs,
             'kl_mask': kl_masks,
             'kl_latent': kl_zs,
             'z': zs,
             'mask': masks,
             'slot': slots,
             'mask_pred': masks_pred,
             'log_mask_pred': log_masks_pred.unsqueeze(2),
             'log_mask': log_masks.unsqueeze(2)}

    def forward_vae_slots(self, log_masks, slots, x):
        masks_pred = []
        neg_log_p_xs = []
        kl_zs = 0.0
        num_slots = log_masks.shape[1]
        zs = torch.zeros(x.size(0), num_slots, self.latent_size)
        for i in range(num_slots):
            log_mask = log_masks[:, i].unsqueeze(1)
            z, kl_z = self._encode(x, log_mask)
            sigma = self.bg_sigma if i == 0 else self.fg_sigma
            neg_log_p_x_masked, x_recon, mask_pred = self._decode(x, z, log_mask, sigma)
            neg_log_p_xs.append(neg_log_p_x_masked.unsqueeze(1))
            kl_zs += kl_z.mean(dim=0)
            masks_pred.append(mask_pred)
            slots[:, i] = x_recon
            zs[:, i, :] = z
        return kl_zs, masks_pred, neg_log_p_xs, zs

    def _calc_loss(self, kl_zs, masks, masks_pred, neg_log_p_xs):
        loss = 0.0
        neg_log_p_xs = - neg_log_p_xs.logsumexp(dim=1).mean(dim=0).sum()
        loss += neg_log_p_xs + self.beta_kl * kl_zs
        kl_masks = self._calc_kl_mask(masks, masks_pred)
        loss += self.gamma * kl_masks
        return kl_masks, loss, neg_log_p_xs

    def _calc_kl_mask(self, masks, masks_pred):
        bs = len(masks)
        flat_masks = masks.permute(0, 2, 3, 1)
        nrows = prod(flat_masks.shape[:-1])
        flat_masks = flat_masks.reshape(nrows, -1)
        flat_masks_pred = masks_pred.permute(0, 2, 3, 1)
        flat_masks_pred = flat_masks_pred.reshape(nrows, -1)
        flat_masks = flat_masks.clamp_min(1e-5)
        flat_masks_pred = flat_masks_pred.clamp_min(1e-5)
        d_masks = dists.Categorical(probs=flat_masks)
        d_masks_pred = dists.Categorical(probs=flat_masks_pred)
        kl_masks = dists.kl_divergence(d_masks, d_masks_pred)
        kl_masks = kl_masks.sum() / bs
        return kl_masks

    def _encode(self, x, log_mask):
        encoder_input = torch.cat((x, log_mask), 1)
        q_params = self.encoder(encoder_input)
        means = q_params[:, :self.latent_size]
        sigmas = F.softplus(q_params[:, self.latent_size:])
        latent_normal = dists.Normal(means, sigmas)
        kl_z = dists.kl_divergence(latent_normal, self.prior_dist)
        kl_z = kl_z.sum(dim=1)
        z = means + sigmas * torch.randn_like(sigmas)
        return z, kl_z

    def _decode(self, x, z, log_mask, sigma):
        decoder_output = self.decoder(z)
        x_recon = decoder_output[:, :3].sigmoid()
        mask_pred = decoder_output[:, 3].unsqueeze(1)
        dist = dists.Normal(x_recon, sigma)
        neg_log_p_x_masked = log_mask + dist.log_prob(x)
        return neg_log_p_x_masked, x_recon, mask_pred

    @classmethod
    def from_config(cls, model: Literal['monet', 'monet-iodine', 'monet-lightweight'] = 'monet',
               dataset: Optional[Literal['clevr_6', 'multidsprites_colored_on_grayscale',
                                         'tetrominoes', 'multidsprites_colored_on_colored']] = None,
               scene_max_objects: int = 5, dataset_width: int = 64, dataset_height: int = 64) -> 'Monet':

        assert model in _MODEL_CONFIG_VALUES

        monet_config = OmegaConf.load(CONFIG_MODEL / f'{model}.yaml')
        if dataset is None:
            monet_config.merge_with(OmegaConf.from_dotlist([
                f"dataset.width={dataset_width}",
                f"dataset.height={dataset_height}",
                f"dataset.max_num_objects={scene_max_objects}"
            ]))
        else:
            assert dataset in _DATASET_CONFIG_VALUES
            monet_config.merge_with(OmegaConf.load(CONFIG_DATASET / f'{dataset}.yaml'))
        if dataset is not None:
            assert dataset in _DATASET_CONFIG_VALUES
            special_case_path = CONFIG_SPECIAL_CASES / f'{model}-{dataset}.yaml'
            if special_case_path.exists():
                monet_config.merge_with(OmegaConf.load(special_case_path))

        return hydra.utils.instantiate(monet_config.model)

    @classmethod
    def from_custom_config(cls, monet_config: DictConfig):
        target = monet_config.model['_target_']
        assert target.startswith('monet_pytorch') and target.endswith('Monet')
        return hydra.utils.instantiate(monet_config.model)
