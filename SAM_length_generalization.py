import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics

import jax
import jax.numpy as jnp
from jax import jit

from pytorch_lightning.strategies import DDPStrategy

from paper_utils import LightningResNet18

import configparser
from paper_utils import PoincareDataModule

def main(args):
    isBool = lambda x: x.lower() == "true"
    converters = {'IntList': lambda x: [int(i.strip()) for i in x.strip(" [](){}").split(',')],
        'FloatList': lambda x: [float(i.strip()) for i in x.strip(" [](){}").split(',')],
        'BoolList': lambda x: [isBool(i.strip()) for i in x.strip(" [](){}").split(',')]}
    config = configparser.ConfigParser(converters=converters)
    config.read(args.cfg)
    # LOGGING params
    # -------------------------------------------------------
    log_dir = config.get('LOGGING', 'log_dir')
    name = config.get('LOGGING', 'name')
    # -------------------------------------------------------
    # HARDWARE params  
    # -------------------------------------------------------  
    num_workers = config.getint('HARDWARE', 'num_workers')
    auto_select_gpus = config.getboolean('HARDWARE', 'auto_select_gpus')
    num_gpus = config.getint('HARDWARE', 'num_gpus')

    if auto_select_gpus==True:
        devices = num_gpus
    else:
        devices = config.getIntList("HARDWARE", "devices")
    if devices > 1:
        num_workers = 0
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        num_workers = num_workers
        strategy = None
    # -------------------------------------------------------
    # DATAMODULE params
    # -------------------------------------------------------
    data_dir = config.get('DATAMODULE', 'data_dir')
    main_lookup_dir = config.get('DATAMODULE', 'main_lookup_dir')
    local_lookup_dir = config.get('DATAMODULE', 'local_lookup_dir')
    batch_size = config.getint('DATAMODULE', 'batch_size')
    img_widths = config.getIntList('DATAMODULE', 'img_width')
    alpha = config.getfloat('DATAMODULE', 'alpha')
    min_samples = config.getint('DATAMODULE', 'min_samples', fallback=0)
    max_samples = config.getint('DATAMODULE', 'max_samples', fallback=-1)
    min_traj_len = config.getint('DATAMODULE', 'min_traj_len')
    max_traj_len = config.getint('DATAMODULE', 'max_traj_len')
    num_params = config.getint('DATAMODULE', 'num_params')
    if ("param_min" in config['DATAMODULE']) or ("param_max" in config['DATAMODULE']):
        if not (("param_min" in config['DATAMODULE']) and ("param_max" in config['DATAMODULE'])):
            raise ValueError("Must specify both param_min and param_max, or neither")
        else:
            param_min = np.array(config.getFloatList('DATAMODULE', 'param_min'))
            param_max = np.array(config.getFloatList('DATAMODULE', 'param_max'))
            assert(num_params == len(param_min) == len(param_max))
    else:
        param_min = None
        param_max = None

    coords = config.getIntList('DATAMODULE', 'coords', fallback=[0,1])
    print(f"{coords=}")
    x_range = config.getFloatList('DATAMODULE', 'x_range', fallback=None)
    y_range = config.getFloatList('DATAMODULE', 'y_range', fallback=None)