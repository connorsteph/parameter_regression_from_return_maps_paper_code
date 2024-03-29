import numpy as np
import torch
import pytorch_lightning as pl
import configparser
import argparse
import pandas as pd
from pathlib import Path

from pytorch_lightning.strategies import DDPStrategy
from paper_utils import PoincareDataModule, LightningResNet18


def main(args):
    def isBool(x): return x.lower() == "true"
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

    if auto_select_gpus == True:
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
    main_lookup_dir = config.get('DATAMODULE', 'main_lookup_dir')
    local_lookup_dir = config.get('DATAMODULE', 'local_lookup_dir')
    sample_fracs = config.getFloatList('DATAMODULE', 'sample_frac')
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
            raise ValueError(
                "Must specify both param_min and param_max, or neither")
        else:
            param_min = np.array(
                config.getFloatList('DATAMODULE', 'param_min'))
            param_max = np.array(
                config.getFloatList('DATAMODULE', 'param_max'))
            assert (num_params == len(param_min) == len(param_max))
    else:
        param_min = None
        param_max = None

    coords = config.getIntList('DATAMODULE', 'coords', fallback=[0, 1])
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
    max_train_epochs = config.getint(
        'TRAINER', 'max_train_epochs', fallback=-1)
    max_train_steps = config.getint('TRAINER', 'max_train_steps', fallback=-1)
    max_train_time = config.get(
        'TRAINER', 'max_train_time', fallback="01:00:00:00")
    print(f'{max_train_time=}, {max_train_epochs=}, {max_train_steps=}')
    top_k_models = config.getint('TRAINER', 'top_k', fallback=1)
    num_runs = config.getint('TRAINER', 'num_runs', fallback=1)
    # -------------------------------------------------------

    if args.split_data == True:
        Path(local_lookup_dir).mkdir(parents=True, exist_ok=True)
        # check if local lookup table exists already
        if Path.exists(Path(local_lookup_dir + "lookup_train.csv")):
            # ask user if they want to overwrite
            overwrite = input(
                f"You passed `--split_data` but lookup tables already exists at {local_lookup_dir}. Overwrite? (y/n): ")
            if overwrite.upper() in ["Y", "YES"]:
                print("Overwriting...")
                pass
            else:
                print("Exiting...")
                exit()
        else:
            pass

        print("Creating train/test/val splits...")

        train_list = []
        test_list = []
        val_list = []
        # perform train/test/val split on the main lookup table, after random shuffling
        df = pd.read_csv(main_lookup_dir + "main_lookup.csv",
                         delimiter=",").sample(frac=1., random_state=np.random.RandomState(42))
        n_total = len(df)
        n_train = int(.8*n_total)
        n_val = n_total - n_train
        n_train = int(0.8*n_train)
        n_test = n_total - n_val - n_train
        train_list += df.iloc[:n_train].values.tolist()
        test_list += df.iloc[n_train:n_train+n_test].values.tolist()
        val_list += df.iloc[n_train+n_test:].values.tolist()

        columns = df.columns
        train_df = pd.DataFrame(train_list, columns=columns)
        test_df = pd.DataFrame(test_list, columns=columns)
        val_df = pd.DataFrame(val_list, columns=columns)
        train_df.to_csv(local_lookup_dir + "lookup_train.csv",
                        sep=",", index=False)
        test_df.to_csv(local_lookup_dir + "lookup_test.csv",
                       sep=",", index=False)
        val_df.to_csv(local_lookup_dir + "lookup_val.csv",
                      sep=",", index=False)
        print("Done. Lookup tables saved to", local_lookup_dir)

    if param_min and param_max:
        # set parameters for re-weighted MSE
        param_widths = param_max - param_min
        param_midpts = (param_max - param_min)/2 + param_min
        # the offsets aren't really needed, but we keep them to center the adjusted parameter range around 0
        loss_offsets = param_midpts.tolist()
        # shift and scale factor to put model outputs / labels in [-5,5] range
        loss_weights = (10./param_widths).tolist()
    else:
        loss_weights = [1. for _ in range(num_params)]
        loss_offsets = [0. for _ in range(num_params)]

    for img_width in img_widths:
        for randomize in randomize_flags:
            for sample_frac in sample_fracs:
                # reset random seeds to ensure that order of run execution doesn't matter
                np.random.seed(42)
                torch.manual_seed(42)

                datamodule = PoincareDataModule(
                    local_lookup_dir=local_lookup_dir,
                    data_dir=data_dir,
                    coords=coords,
                    sample_frac=sample_frac,
                    img_width=img_width,
                    batch_size=batch_size,
                    x_range=x_range,
                    y_range=y_range,
                    alpha=alpha,
                    min_samples=min_samples,
                    max_samples=max_samples,
                    min_traj_len=min_traj_len,
                    max_traj_len=max_traj_len,
                    randomize=randomize,
                    num_workers=num_workers,
                    verbose=True
                )
                for run in range(num_runs):
                    print(f"Starting run {run+1} of {num_runs}")
                    logger = pl.loggers.TensorBoardLogger(
                        log_dir+f'/sample_frac_{sample_frac:2.2f}/img_width_{img_width}/randomized_{randomize}/', name + f"_sample_frac_{sample_frac:2.2f}")
                    checkpoint_callback = pl.callbacks.ModelCheckpoint(
                        monitor='val_mse',
                        filename='{epoch:02d}-{val_mse:1.2e}' +
                        f'_randomized_{randomize}',
                        save_top_k=top_k_models,
                        mode='min',
                    )

                    datamodule.logger = logger
                    datamodule.log_params()

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
                        strategy=strategy,
                        deterministic=deterministic,
                        devices=devices,
                        precision=precision,
                        max_steps=max_train_steps,
                        max_epochs=max_train_epochs,
                        max_time=max_train_time,
                    )

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
    parser.add_argument('--split_data', action='store_true')
    main(parser.parse_args())
