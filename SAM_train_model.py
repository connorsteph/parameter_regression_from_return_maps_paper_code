import numpy as np
import torch
import pytorch_lightning as pl
import configparser
import argparse

from pytorch_lightning.strategies import DDPStrategy
from paper_utils import PoincareDataModule, LightningResNet18

np.random.seed(42)
torch.manual_seed(42)

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
    root_dir = config.get('LOGGING', 'root_dir')
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
    data_dir = config.get('DATAMODULE', 'data_dir')
    local_lookup_dir = config.get('DATAMODULE', 'local_lookup_dir')
    sample_fracs = config.getFloatList('DATAMODULE', 'sample_frac')
    batch_size = config.getint('DATAMODULE', 'batch_size')
    img_widths = config.getIntList('DATAMODULE', 'img_width')
    alpha = config.getfloat('DATAMODULE', 'alpha')
    min_samples = config.getint('DATAMODULE', 'min_samples')
    min_traj_len = config.getint('DATAMODULE', 'min_traj_len')
    max_traj_len = config.getint('DATAMODULE', 'max_traj_len')
    mu_min = config.getfloat('DATAMODULE', 'mu_min')
    mu_max = config.getfloat('DATAMODULE', 'mu_max')
    # -------------------------------------------------------
    # TRAINER params
    # -------------------------------------------------------
    learning_rate = config.getfloat('TRAINER', 'learning_rate')
    weight_decay = config.getfloat('TRAINER', 'weight_decay')
    deterministic = config.getboolean('TRAINER', 'deterministic')
    precision = config.getint('TRAINER', 'precision')
    if "max_train_epochs" in config['TRAINER']:
        max_train_epochs = config.getint('TRAINER', 'max_train_epochs')
    else:
        max_train_epochs = -1
    if "max_train_steps" in config['TRAINER']:
        max_train_steps = config.getint('TRAINER', 'max_train_steps')
    else:
        max_train_steps = -1
    if "max_train_time" in config['TRAINER']:
        max_train_time = config.get('TRAINER', 'max_train_time')
    else:
        max_train_time = "01:00:00:00"
    print(max_train_time, max_train_steps, max_train_epochs)
    top_k = config.getint('TRAINER', 'top_k')
    # ------------------------------------------------------- 
    for img_width in img_widths:
        for randomize in randomize_flags:
            for sample_frac in sample_fracs:
                logger = pl.loggers.TensorBoardLogger(log_dir+f'/img_width_{img_width}/randomized_{randomize}/', name + f"_sample_frac_{sample_frac:2.2f}")
                checkpoint_callback = pl.callbacks.ModelCheckpoint(
                        monitor='val_mse',
                        filename='{epoch:02d}-{val_mse:1.2e}' + f'_randomized_{randomize}',
                        save_top_k=top_k,
                        mode='min',
                    )
                    
                datamodule = PoincareDataModule(
                    local_lookup_dir=local_lookup_dir,
                    data_dir=data_dir,
                    sample_frac=sample_frac,
                    img_width=img_width,
                    batch_size=batch_size, 
                    alpha=alpha,
                    min_samples=min_samples,
                    min_traj_len=min_traj_len,
                    max_traj_len=max_traj_len,
                    randomize=randomize,
                    num_workers=num_workers,
                    logger=logger,
                    verbose=True
                )

                trainer = pl.Trainer(
                    logger=logger, 
                    default_root_dir=root_dir,
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
                mu_width = mu_max - mu_min
                mu_mid = (mu_max - mu_min)/2 + mu_min
                # the offsets aren't really needed, but we keep them to center the adjusted parameter range around 0
                loss_offsets = [mu_mid]
                loss_weights = [10/mu_width,] # shift and scale factor to put model outputs / labels in [-5,5] range
                model = LightningResNet18(out_dim=1,
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