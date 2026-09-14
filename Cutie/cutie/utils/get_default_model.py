"""
A helper function to get a default model for quick testing
"""
import os
from omegaconf import open_dict
from hydra import compose, initialize

import torch
from cutie.model.cutie import CUTIE
from cutie.inference.utils.args_utils import get_dataset_cfg
from cutie.utils.download_models import download_models_if_needed


def get_default_model() -> CUTIE:
    # ``get_default_model`` is called once per camera.  Use Hydra's context
    # manager so its process-global state is released after each composition;
    # otherwise constructing the second stereo tracker fails with "GlobalHydra
    # is already initialized".
    with initialize(version_base='1.3.2', config_path="../config", job_name="eval_config"):
        cfg = compose(config_name="eval_config")

    weight_dir = download_models_if_needed()
    with open_dict(cfg):
        cfg['weights'] = os.path.join(weight_dir, 'cutie-base-mega.pth')
    get_dataset_cfg(cfg)

    # Load the network weights
    cutie = CUTIE(cfg).cuda().eval()
    # Keep checkpoint deserialization on CPU. ``CUTIE`` has already been moved
    # to CUDA above, and ``load_weights``/``load_state_dict`` safely copy the
    # CPU tensors into that model. This avoids legacy CUDA-storage failures in
    # recent PyTorch releases.
    model_weights = torch.load(cfg.weights, map_location="cpu")
    cutie.load_weights(model_weights)

    return cutie
