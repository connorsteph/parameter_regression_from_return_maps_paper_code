import numpy as np
import pandas as pd

from os import path

import jax
import jax.numpy as jnp
from jax import jit

import pytorch_lightning as pl
import torchmetrics 
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn.functional as F

def randint(low, high):
    # extends np.random.randint to handle degenerate cases
    if low >= high:
        if low == high:
            return low
        else:
            raise ValueError(f"{low=} > {high=}")
    return np.random.randint(low, high)

def henon_param_pixel_map(
    ab_pair,
    img_width,
    num_pts=None,   
    dim_pts=15,
    iters=500,
    alpha=.7,
    x_range=[-4,4],
    y_range=[-4,4],
    random_crop=False,
    min_samples=1,
    min_traj_length=10,
):
    tf = MapListToPixelArray(img_width=img_width, x_range=x_range, y_range=y_range, alpha=alpha)
    with jax.default_device(jax.devices('cpu')[0]):  
        @jit
        def generate_map(ab_pair):
            nonlocal dim_pts
            a,b = ab_pair
            if dim_pts is None:
                dim_pts = int(np.round(np.sqrt(num_pts)))
            X = jnp.array(jnp.meshgrid(jnp.linspace(*x_range, dim_pts), jnp.linspace(*y_range, dim_pts))).reshape(2, -1).T
            return batched_henon_map_collect(X, iters=iters, a=a, b=b).transpose(0,2,1)

        map_list = generate_map(ab_pair)
        if random_crop == True:
            return transforms.Compose([RandomTrajectoryCrop(min_length=min_traj_length), RandomCrop(min_length=min_samples), tf])({"map_list": map_list, "label": ab_pair})['image']
        return tf({"map_list": map_list, "label": ab_pair})['image']


class RandomTrajectoryCrop(object):
    def __init__(
        self, min_length=None
    ):
        self.min_length = min_length

    def __call__(self, sample):
        if sample['map_list'].dtype == object: # ragged map list
            map_list = sample['map_list']
            if self.min_length:
                map_list = np.array([traj[:randint(self.min_length, len(traj))] for traj in map_list],dtype=object)
            cropped_len = -1
        else:
            if self.min_length:
                cropped_len = randint(self.min_length, len(sample['map_list'][0])) # assumes a uniform length for all traj
            else:
                cropped_len = len(sample['map_list'][0])
            map_list = sample['map_list'][:,:cropped_len,:] # crop trajectories to a random length
        
        return {"label": sample['label'], "map_list": map_list, "cropped_len": cropped_len}


class RandomCrop(object):
    def __init__(
    self, min_length=10
    ):
        self.min_length = min_length

    def __call__(self, sample):
        map_list = sample['map_list']
        num_traj = randint(self.min_length, len(map_list))
        map_list = np.random.permutation(map_list)[:num_traj]  # select a random subset of trajectories
        return {"label": sample['label'], "map_list": map_list, "cropped_len": sample["cropped_len"], "num_traj": num_traj}
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
class MapListToPixelArray(object):
    """Converts a numpy array of Poincare section crossing coordinates to an image array of specified width,

    Args:
        img_width (int): Desired width of image
        alpha (float, optional): alpha value for pixel darkening per hit
        x_range (tuple, optional): range of x values to be mapped to image
        y_range (tuple, optional): range of y values to be mapped to image
    """

    def __init__(
        self,
        img_width,
        x_range=None,
        y_range=None,
        alpha=None,
    ):
        self.img_width = img_width
        self.x_range=x_range
        self.y_range=y_range
        self.alpha=alpha

    def __call__(self, sample):
        map_list = sample["map_list"]
        pixel_array = np.ones(
            (self.img_width, self.img_width), dtype=np.float16
        )
        x_lims = [
                np.min([np.min(trajs[:, 0]) for trajs in map_list]),
                np.max([np.max(trajs[:, 0]) for trajs in map_list]),]
        y_lims = [
                np.min([np.min(trajs[:, 1]) for trajs in map_list]),
                np.max([np.max(trajs[:, 1]) for trajs in map_list]),]

        for trajs in map_list:
            # remove all items in map with coord 0 outside of x_range and coord 1 outside of y_range, if they exist
            indices = np.ones(len(trajs), dtype=bool)
            if self.x_range:
                indices *= np.greater_equal(trajs[:,0], self.x_range[0]) \
                * np.less_equal(trajs[:,0], self.x_range[1])
            if self.y_range:
                indices *= np.greater_equal(trajs[:,1], self.y_range[0]) \
                * np.less_equal(trajs[:,1], self.y_range[1])
            trajs = trajs[indices, :].astype(np.float16)

            # rescale x and y values to be in [0,1] range, either shifting and scaling to center and stretch the resulting image, or 
            # using the values of x_range, y_range if provided
            if self.x_range:
                # scale Poincare map to fit into x_range
                x_scaled = (trajs[:, 0] - self.x_range[0]) / (
                    self.x_range[1] - self.x_range[0]
                )  # scale x_vals to (0,1) range
            else: 
                # shift and scale into unit square
                x_scaled = (trajs[:, 0] - x_lims[0]) / (
                    x_lims[1] - x_lims[0]
                )  # scale x_vals to (0,1) range
            
            if self.y_range:
                # scale Poincare map to fit into y_range
                y_scaled = (trajs[:, 1] - self.y_range[0]) / (
                    self.y_range[1] - self.y_range[0]
                )  # scale y_vals to (0,1) range
            else:
                # shift and scale into unit square
                y_scaled = (trajs[:, 1] - y_lims[0]) / (
                    y_lims[1] - y_lims[0]
                )  # scale y_vals to (0,1) range

            # discretize crossings to grid coordinates
            x_scaled = np.uint8(np.round((self.img_width - 1) * x_scaled))
            y_scaled = np.uint8(np.round((self.img_width - 1) * (1.0 - y_scaled)))
            # shade pixels with alpha by counting crossings per square
            pixel_array[y_scaled, x_scaled] *= self.alpha

        image = 2.0 * (pixel_array)
        image -= 1.0
        out = sample.copy()
        out.pop("map_list", None)
        out["image"] = image
        return out
#--------------------------------------------------------------------------------------------------
class ToTensor(object):
    def __call__(self, sample):
        image = torch.tensor(sample['image'][np.newaxis, ...],dtype=torch.float)
        out = {"image":image}
        for key in sample.keys():
            if key not in ["image"]:
                out[key] = torch.tensor(sample[key])
        return out
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
class custom_collate_fn:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, batch):
        transformed_batch = [self.transform(item) for item in batch]
        return {key: torch.stack([item[key] for item in transformed_batch]) for key in transformed_batch[0].keys()}
#--------------------------------------------------------------------------------------------------
def build_datasets(
        param_samples, img_width, iters=500,
        grid_dim_pts = 15, grid_samples=None,
        x_range = [-4,4], y_range = [-4,4],
        min_length=10,
        min_traj_len=None,
        a_range = [0.05, .45],
        b_range = [-1.1, 1.1],
        alpha=.7,
        randomize=False,
        verbose=False
    ):
    with jax.default_device(jax.devices('cpu')[0]):  
        param_dim_pts = int(np.round(np.sqrt(param_samples)))
        a_vals = jnp.linspace(*a_range, param_dim_pts)
        b_vals = jnp.linspace(*b_range, param_dim_pts)
        @jit
        def generate_map(ab_pair):
            nonlocal grid_dim_pts
            a,b = ab_pair
            if grid_dim_pts is None:
                grid_dim_pts = int(np.round(np.sqrt(grid_samples)))
            X = jnp.array(jnp.meshgrid(jnp.linspace(*x_range, grid_dim_pts), jnp.linspace(*y_range, grid_dim_pts))).reshape(2, -1).T
            return batched_henon_map_collect(X, iters=iters, a=a, b=b).transpose(0,2,1)
        if verbose:
            print(f"Generating dataset with {param_samples} total samples")
        dataset = []
        a_b_pairs = jnp.array(jnp.meshgrid(a_vals, b_vals)).reshape(2, -1).T
        for ab_pair, map_list in zip(a_b_pairs, map(generate_map, a_b_pairs)):
            a,b = ab_pair
            dataset.append({"label":np.array([a,b]), "map_list": np.array(map_list)})
        dataset = np.array(dataset)
        np.random.shuffle(dataset)
        if verbose:
            print(f"Done.")
        if randomize==True:
            tf = transforms.Compose([RandomTrajectoryCrop(min_length=min_traj_len), RandomCrop(min_length=min_length), MapListToPixelArray(img_width=img_width, alpha=alpha, 
                x_range=x_range, y_range=y_range)])
        else:
            tf = MapListToPixelArray(img_width=img_width, alpha=alpha, 
                x_range=x_range, y_range=y_range)
        if verbose:
            print("Performing split...")

        train_data = dataset[:int(.64*len(dataset))]
        # apply data augmentation to val and test sets once
        val_data = dataset[int(.64*len(dataset)):int(.8*len(dataset))]
        val_data = np.array([tf(item) for item in val_data])
        test_data = dataset[int(.8*len(dataset)):]
        test_data = np.array([tf(item) for item in test_data])
        if verbose:
            print(f"Done. \nTrain: {train_data.shape[0]}, Val: {val_data.shape[0]}, Test: {test_data.shape[0]}")
        return train_data, val_data, test_data
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
class HenonMapPoincareDataModule(pl.LightningDataModule):
    def __init__(self, num_samples, img_width, x_range, y_range, 
            a_range=[0.05, .45], b_range=[-1.3, 1.3],
            batch_size=32, num_workers=0,
            num_iters=500, min_samples=10,
            max_samples=225, min_traj_len=1,
            alpha=.7, randomize=True,
            persistent_workers=True,
            verbose=False,
            logger=None
            ):
        super().__init__()
        self.img_width = img_width
        self.x_range = x_range
        self.y_range = y_range
        self.batch_size = batch_size
        self.a_range = a_range
        self.b_range = b_range
        self.min_traj_len = min_traj_len
        self.min_samples=min_samples
        self.num_iters = num_iters
        self.max_samples = max_samples
        self.randomize = randomize
        self.persistent_workers = persistent_workers
        self.num_workers = num_workers
        self.logger = logger
        if verbose:
            print("Building datasets...")
            if randomize==True:
                print("Using data augmentation")
            else:
                print("Not using data augmentation")
        self.train_data, self.val_data, self.test_data = build_datasets(num_samples, 
            img_width, x_range=x_range, y_range=y_range,
            a_range=a_range, b_range=b_range,
            alpha=alpha,
            grid_dim_pts=int(np.round(np.sqrt(max_samples))),
            min_length=min_samples, iters=num_iters, min_traj_len=min_traj_len,
            randomize=randomize, verbose=verbose)
        if verbose:
            print("Built")
        if randomize==True:
            self.transform = transforms.Compose([RandomTrajectoryCrop(min_length=min_traj_len), RandomCrop(min_length=min_samples), MapListToPixelArray(img_width=img_width, alpha=alpha, 
                x_range=x_range, y_range=y_range),ToTensor()])
        else:
            self.transform = transforms.Compose([MapListToPixelArray(img_width=img_width, alpha=alpha, 
                x_range=x_range, y_range=y_range), ToTensor()])
        if logger is not None:
            self.logger.log_hyperparams({"num_samples": num_samples, "img_width":img_width, "x_range":x_range,
            "y_range":y_range, "batch_size":batch_size, "a_range":a_range, "b_range":b_range,
            "min_traj_len":min_traj_len, "min_samples": min_samples, "num_iters": num_iters,
            "max_samples":max_samples, "randomize":randomize, "num_train": len(self.train_data),
            "num_val": len(self.val_data), "num_test": len(self.test_data), "alpha":alpha,})

    def set_batch_size(self, batch_size):
        if self.logger is not None:
            self.logger.log_hyperparams({"batch_size":batch_size})
        self.batch_size = batch_size

    def get_batch_size(self):
        return self.batch_size

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size,
         collate_fn = custom_collate_fn(self.transform), shuffle=True, pin_memory=True, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
        # return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers, persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,
         collate_fn = custom_collate_fn(ToTensor()), shuffle=False, pin_memory=True, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
        # return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size,
         collate_fn = custom_collate_fn(ToTensor()), shuffle=False, pin_memory=True, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
        # return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
class PoincareDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        local_lookup_dir,
        data_dir,
        coords=None,
        sample_frac=1.0,
        img_width=128,
        batch_size=32, num_workers=0,
        min_samples=10,
        max_samples=None,
        min_traj_len=1,
        max_traj_len=250,
        x_range=None,
        y_range=None,
        alpha=.7, randomize=True,
        persistent_workers=True,
        pin_memory=True,
        verbose=False,
        logger=None,
        ):
        super().__init__()
        self.img_width = img_width
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_samples = min_samples
        self.coords = coords
        self.min_traj_len=min_traj_len,
        self.sample_frac=sample_frac
        self.alpha = alpha,
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.logger=logger

        if verbose:
            print("Loading dataset:")
            if randomize:
                print("Using data augmentation")
            print("Training set...")
        if randomize:
            self.transform = transforms.Compose([RandomTrajectoryCrop(min_length=min_traj_len), 
            RandomCrop(min_length=min_samples), MapListToPixelArray(img_width=img_width, alpha=alpha, x_range=x_range, y_range=y_range), ToTensor()])
            pre_transform = transforms.Compose([RandomTrajectoryCrop(min_length=min_traj_len), 
            RandomCrop(min_length=min_samples), MapListToPixelArray(img_width=img_width, alpha=alpha, x_range=x_range, y_range=y_range)])
        else:
            self.transform = transforms.Compose([MapListToPixelArray(img_width=img_width, alpha=alpha, x_range=x_range, y_range=y_range), ToTensor()])
            pre_transform = transforms.Compose([MapListToPixelArray(img_width=img_width, alpha=alpha, x_range=x_range, y_range=y_range)])
        self.train_data = LoadPoincareMapGridDatasetInMemory(local_lookup_dir, "lookup_train.csv",
            data_dir, coord_indices=self.coords, sample_frac=sample_frac, max_traj_len=max_traj_len,
            max_samples=max_samples, verbose=verbose)
        if verbose:
            print("Validation set...")
        self.val_data = LoadPoincareMapGridDatasetInMemory(local_lookup_dir, "lookup_val.csv",
            data_dir, coord_indices=self.coords, sample_frac=sample_frac, transform=pre_transform, max_traj_len=max_traj_len,
            max_samples=max_samples, verbose=verbose)
        if verbose:
            print("Test set...")
        self.test_data = LoadPoincareMapGridDatasetInMemory(local_lookup_dir, "lookup_test.csv",
            data_dir, coord_indices=self.coords, sample_frac=sample_frac, transform=pre_transform, max_traj_len=max_traj_len,
            max_samples=max_samples, verbose=verbose)
        if verbose:
            print("Dataset loaded.")
        if logger is not None:
            self.logger.log_hyperparams({"sample_frac": sample_frac, "img_width":img_width, "x_range":x_range,
            "y_range":y_range, "batch_size":batch_size, "min_traj_len":min_traj_len, "min_samples": min_samples,
            "max_samples":max_samples, "randomize":randomize, "num_train": len(self.train_data),
            "num_val": len(self.val_data), "num_test": len(self.test_data), "alpha":alpha, "coords":coords,})

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_batch_size(self):
        return self.batch_size

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size,
         collate_fn = custom_collate_fn(self.transform), shuffle=True, pin_memory=self.pin_memory, num_workers=self.num_workers, persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,
         collate_fn = custom_collate_fn(ToTensor()), shuffle=False, pin_memory=self.pin_memory, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size,
         collate_fn = custom_collate_fn(ToTensor()), shuffle=False, pin_memory=self.pin_memory, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
class LoadPoincareMapGridDatasetInMemory(Dataset):
    def __init__(
        self,
        lookup_dir,
        lookup_fname,
        data_dir,
        max_traj_len=None,
        max_samples=None,
        sample_frac=None,
        transform=None,
        coord_indices=None,
        verbose=False
    ):
        """
        Args:
            root_dir (string): Path to find folders containing lookup files and data folders of .npz files
            lookup_file (string): name of the .csv file containing the lookup table of file names for the samples
            transform (callable, optional): Optional transform to be applied on a sample
        """
        # sanitize lookup_df
        self.lookup_df = pd.read_csv(lookup_dir + lookup_fname, delimiter=",")
        if sample_frac:
            assert(0 < sample_frac <= 1.0)
            self.lookup_df = self.lookup_df.sample(frac=sample_frac)
        is_file = lambda x: path.isfile(data_dir + x)
        is_file = np.vectorize(is_file)
        self.lookup_df = self.lookup_df[is_file(self.lookup_df['fname'])]
        self.data_dir = data_dir
        self.transform = transform
        self.coord_indices = coord_indices
        self.entries = np.empty(len(self.lookup_df), dtype=object)
        for i in range(len(self.lookup_df)):
            if verbose:
                print(f"{i:3d} / {len(self.lookup_df):3d}", end="\r")
            fname = self.lookup_df.iloc[i]["fname"]
            if path.isfile(
                self.data_dir + fname
            ):
                curr_npz = np.load(self.data_dir + fname, allow_pickle=True)
                trajs = curr_npz["coord_traj"]
                label = self.lookup_df.iloc[i]["mu"]
                if isinstance(label, np.ndarray):
                    label = label.astype(np.float16)
                else:
                    label = np.array([label], dtype=np.float16)
                if max_samples is not None and max_samples > 0:
                    np.random.shuffle(trajs)
                    trajs = trajs[:max_samples]
                if max_traj_len is not None and max_traj_len > 0:
                    trajs = np.array([traj[:min(max_traj_len, len(traj)), self.coord_indices] for traj in trajs], dtype=object)
                sample = {
                    "label": label,
                    "map_list": trajs,
                }
                if self.transform:
                    sample = self.transform(sample)
                self.entries[i] = sample
        self.length = len(self.entries)
        if self.length == 0:
            raise ValueError("No entries found")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.entries[idx]
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

class LightningResNet18(pl.LightningModule):
    def __init__(self, out_dim=2, learning_rate=1e-5, weight_decay=0., loss_weights=None, loss_offsets=None):
        super().__init__()
        self.save_hyperparameters()
        self.out_dim = out_dim
        # resnet18 modified for single channel images
        self.model = resnet18(weights=False, num_classes=out_dim)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.best_val_mse = torch.inf
        self.train_metric = torchmetrics.MeanSquaredError(compute_on_step=False)
        self.valid_metric = torchmetrics.MeanSquaredError(compute_on_step=False)
        if loss_weights is not None:
            self.loss_weights = torch.tensor(loss_weights, device=self.device)
        else:
            self.loss_weights = torch.ones(out_dim, device=self.device)
        if loss_offsets is not None:
            self.loss_offsets = torch.tensor(loss_offsets, device=self.device)
        else:
            self.loss_offsets = torch.zeros(out_dim, device=self.device)
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 
                                lr=self.hparams.learning_rate, 
                                weight_decay=self.hparams.weight_decay)
    
    def training_step(self, batch, batch_idx):
        # if loss weights on wrong device, move them
        if self.loss_weights.device != self.device:
            self.loss_weights = self.loss_weights.to(self.device)
        if self.loss_offsets.device != self.device:
            self.loss_offsets = self.loss_offsets.to(self.device)
        x, y = batch['image'], self.loss_weights * (self.loss_offsets + batch['label']) # multiply by weights to account for scale factors
        y_hat = self.loss_weights * (self.loss_offsets + self(x))
        loss = F.mse_loss(y_hat, y)
        return {'loss' : loss, 'y_hat' : y_hat, 'y' : y}

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)
      
    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)
    
    def training_step_end(self, outputs):
        loss = outputs['loss'].mean()
        self.train_metric(outputs['y_hat'], outputs['y'])
        self.log("train_mse", loss, sync_dist=True, on_step=False, on_epoch=True)
        return {'loss' : loss}
    
    def validation_step_end(self, outputs):
        loss = outputs['loss'].mean()
        self.valid_metric(outputs['y_hat'], outputs['y'])
        self.log("val_mse", loss, sync_dist=True, on_step=False, on_epoch=True)
        return {'loss' : loss}
    
    def validation_epoch_end(self, outputs):
        loss = self.valid_metric.compute()
        self.best_val_mse = min(self.best_val_mse, loss.item())
        self.log("best_val_mse", self.best_val_mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log("hp_metric", self.best_val_mse, on_step=False, on_epoch=True, sync_dist=True)
        self.valid_metric.reset()
        return {'val_loss' : loss, 'best_val_mse' : self.best_val_mse}
    
    def test_step_end(self, outputs):
        return self.validation_step_end(outputs)
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

def henon_map_collect(xy_pair, iters=1000, a=1.4, b=.3):
    def cond_fun(carry):
        return carry[0] < iters

    def body_fun(carry):
        i = carry[0]
        val = carry[1][:,i]
        carry[1] = carry[1].at[:,i+1].set(jnp.array([1 - a * val[-2]**2 + val[-1], b * val[-2]]))
        carry[0] = i+1
        return carry
    return jax.lax.while_loop(cond_fun, body_fun, [0, jnp.hstack([jnp.array(xy_pair).reshape(2,1), jnp.empty((2, iters))])])[1]

@jit
def batched_henon_map_collect_helper(container, a=1.4, b=.3):
    # container has shape (n, 2, iters+1)
    iters = container.shape[-1]-1
    n = container.shape[0]

    def cond_fun(carry):
        return carry[0] < n

    def body_fun(carry):
        i = carry[0]
        init_pair = carry[1][i,:,0]
        carry[1] = carry[1].at[i,:,:].set(henon_map_collect(init_pair, iters, a, b))
        carry[0] = i+1
        return carry
    return jax.lax.while_loop(cond_fun, body_fun, [0, container])[1]

def batched_henon_map_collect(X, iters=1000, a=1.4, b=.3):
    return batched_henon_map_collect_helper(
        jnp.dstack([X[:,:,jnp.newaxis],
        jnp.repeat(
            jnp.empty_like(X[:, :, jnp.newaxis]), repeats=iters, axis=2)
            ]),
        a, b)