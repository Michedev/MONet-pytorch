from math import prod
from typing import Literal, Optional

import hydra.utils
from omegaconf import OmegaConf, omegaconf

from monet_pytorch import Monet
from monet_pytorch.paths import CONFIG_MODEL, CONFIG_SPECIAL_CASES, CONFIG_DATASET

OmegaConf.register_new_resolver('prod', lambda *numbers: int(prod(float(x) for x in numbers)))
OmegaConf.register_new_resolver('add', lambda *numbers: int(sum(float(x) for x in numbers)))

_MODEL_CONFIG_VALUES = ['monet', 'monet-iodine', 'monet-lightweight']
_DATASET_CONFIG_VALUES = ['clevr_6', 'multidsprites_colored_on_grayscale',
                          'tetrominoes', 'multidsprites_colored_on_colored']


def init_monet(model: Literal['monet', 'monet-iodine', 'monet-lightweight'] = 'monet',
               dataset: Optional[Literal['clevr_6', 'multidsprites_colored_on_grayscale',
                                         'tetrominoes', 'multidsprites_colored_on_colored']] = None,
               scene_max_objects: int = 5, dataset_width: int = 64, dataset_height: int = 64) -> Monet:
    assert model in _MODEL_CONFIG_VALUES

    monet_config = OmegaConf.load(CONFIG_MODEL / f'{model}.yaml')
    if dataset is None:
        monet_config.merge_with(OmegaConf.from_dotlist([
            f"dataset.width: {dataset_width}",
            f"dataset.height: {dataset_height}",
            f"dataset.max_num_objects: {scene_max_objects}"
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


def init_monet_custom(monet_config: omegaconf.DictConfig) -> Monet:
    target = monet_config.model['_target_']
    assert target.startswith('monet_pytorch') and target.endswith('Monet')
    return hydra.utils.instantiate(monet_config.model)
