import numpy as np
import torch
import pytorch_lightning as pl
import configparser
import argparse

from pytorch_lightning.strategies import DDPStrategy
from paper_utils import HenonMapPoincareDataModule, LightningResNet18

def main(args):
    isBool = lambda x: x.lower() == "true"
    converters = {'IntList': lambda x: [int(i.strip()) for i in list(filter(None, x.strip(" [](){}").split(',')))],
        'FloatList': lambda x: [float(i.strip()) for i in list(filter(None, x.strip(" [](){}").split(',')))],
        'BoolList': lambda x: [isBool(i.strip()) for i in list(filter(None, x.strip(" [](){}").split(',')))]}
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
    randomize_flags = config.getBoolList('DATAMODULE', 'randomize')
    if "num_train_samples" in config['DATAMODULE']:
        # setting total samples according to 80:20 train/test and train+val/test ratios
        num_sample_list = [int(num / .64) for num in config.getIntList('DATAMODULE', 'num_train_samples')]
        if "num_samples" in config['DATAMODULE']:
            print("WARNING: num_samples and num_train_samples both specified in config. num_samples will be ignored")
    else:
        num_sample_list = config.getFloatList('DATAMODULE', 'num_samples')
    batch_size = config.getint('DATAMODULE', 'batch_size')
    img_widths = config.getIntList('DATAMODULE', 'img_width')
    alpha = config.getfloat('DATAMODULE', 'alpha')
    min_samples = config.getint('DATAMODULE', 'min_samples', fallback=0)
    max_samples = config.getint('DATAMODULE', 'max_samples', fallback=-1)
    min_traj_len = config.getint('DATAMODULE', 'min_traj_len')
    max_traj_len = config.getint('DATAMODULE', 'max_traj_len')
    num_params = config.getint('DATAMODULE', 'num_params')
    a_range = config.getFloatList('DATAMODULE', 'a_range')
    b_range = config.getFloatList('DATAMODULE', 'b_range')

    coords = config.getIntList('DATAMODULE', 'coords', fallback=[0,1])
    print(f"{coords=}")
    x_range = config.getFloatList('DATAMODULE', 'x_range', fallback=None)
    y_range = config.getFloatList('DATAMODULE', 'y_range', fallback=None)
    # -------------------------------------------------------
    # TRAINER params
    # -------------------------------------------------------
    learning_rate = config.getfloat('TRAINER', 'learning_rate')
    weight_decay = config.getfloat('TRAINER', 'weight_decay')
    deterministic = config.getboolean('TRAINER', 'deterministic')
    precision = config.getint('TRAINER', 'precision', fallback=16)
    max_train_epochs = config.getint('TRAINER', 'max_train_epochs', fallback=-1)
    max_train_steps = config.getint('TRAINER', 'max_train_steps', fallback=-1)
    max_train_time = config.get('TRAINER', 'max_train_time', fallback="01:00:00:00")
    print(f'{max_train_time=}, {max_train_epochs=}, {max_train_steps=}')
    top_k = config.getint('TRAINER', 'top_k', fallback=1)
    # ------------------------------------------------------- 
    for img_width in img_widths:
        for randomize in randomize_flags:
            for num_samples in num_sample_list:
                # reset random seeds to ensure that order of execution of model training doesn't matter
                np.random.seed(42)
                torch.manual_seed(42)
                
                logger = pl.loggers.TensorBoardLogger(log_dir+f'/num_samples_{num_samples}/img_width_{img_width}/randomized_{randomize}/', name + f"num_samples_{num_samples}")
                checkpoint_callback = pl.callbacks.ModelCheckpoint(
                        monitor='val_mse',
                        filename='{epoch:02d}-{val_mse:1.2e}' + f'_randomized_{randomize}',
                        save_top_k=top_k,
                        mode='min',
                    )
                    
                datamodule = HenonMapPoincareDataModule(
                    num_samples=num_samples,
                    batch_size=batch_size, 
                    a_range=a_range,
                    b_range=b_range,
                    x_range=x_range,
                    y_range=y_range,
                    img_width=img_width,
                    alpha=alpha,
                    min_samples=min_samples,
                    max_samples=max_samples,
                    min_traj_len=min_traj_len,
                    num_iters=max_traj_len,
                    randomize=randomize,
                    num_workers=num_workers,
                    logger=logger,
                    verbose=True
                )

                trainer = pl.Trainer(
                    logger=logger, 
                    log_every_n_steps=20,
                    check_val_every_n_epoch=1,
                    num_sanity_val_steps=0,
                    callbacks=[checkpoint_callback],
                    enable_checkpointing=True,
                    enable_progress_bar=True,
                    auto_select_gpus=auto_select_gpus,
                    accelerator='gpu',
                    strategy = strategy,
                    deterministic=deterministic,
                    devices=devices,
                    precision=precision,
                    max_steps=max_train_steps,
                    max_epochs=max_train_epochs,
                    max_time=max_train_time,
                )


                # set parameters for re-weighted MSE
                # the offsets aren't really needed, but we keep them to center the adjusted parameter range around 0
                a_width = a_range[1] - a_range[0]
                b_width = b_range[1] - b_range[0]
                a_midpt = (a_range[1] + a_range[0]) / 2
                b_midpt = (b_range[1] + b_range[0]) / 2
                loss_offsets = [-a_midpt, -b_midpt]
                loss_weights = [10/a_width, 10/b_width] # shift and scale factor to put model outputs / labels in [-5,5] range
                model = LightningResNet18(out_dim=num_params,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    loss_weights=loss_weights,
                    loss_offsets=loss_offsets
                )
                trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    main(parser.parse_args())