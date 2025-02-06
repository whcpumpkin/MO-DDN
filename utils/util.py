import json
import torch
# import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import errno
import os
import os.path as op
import yaml
import random
import numpy as np
from omegaconf import OmegaConf
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class FixedCategorical(torch.distributions.Categorical):

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1))

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class MaskCategorical(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(MaskCategorical, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, mask=None):
        x = self.linear(x)
        if mask is not None:
            x = x + mask
        return FixedCategorical(logits=x)


def get_valid_transforms(h, w):
    return A.Compose(
        [A.Resize(height=h, width=w, p=1.0), ToTensorV2(p=1.0)],
        p=1.0,
        #bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0, label_fields=['labels'])
    )


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def make_env_fn(args, env_class, rank):
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.
    Args:
        config: root exp config that has core env config node as well as
            env-specific config node.
        env_class: class type of the env to be created.
        rank: rank of env to be created (for seeding).
    Returns:
        env object created according to specification.
    """
    # print(config.SIMULATOR.RGB_SENSOR.HFOV)
    env = env_class(args)
    env.seed(rank)
    return env


def mkdir(path):
    # if it is the current folder, skip.
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_config_file(file_path):
    with open(file_path, 'r') as fp:
        return OmegaConf.load(fp)


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def load_from_yaml_file(yaml_file):
    with open(yaml_file, 'r') as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


def find_file_path_in_yaml(fname, root):
    if fname is not None:
        if op.isfile(fname):
            return fname
        elif op.isfile(op.join(root, fname)):
            return op.join(root, fname)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), op.join(root, fname))


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
