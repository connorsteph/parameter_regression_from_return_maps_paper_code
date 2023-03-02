from os import path
import torch

from torch.utils.data import Dataset
from torchvision import transforms, utils
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np

import pandas as pd

import multiprocessing
import os

import paramiko
from scp import SCPClient


def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client


def pack_pad_collate(batch):
    lengths = [t["traj"].shape[0] for t in batch]
    packed_traj_batch = rnn_utils.pack_sequence(
        [t["traj"] for t in batch], enforce_sorted=False
    )
    packed_label_traj_batch = rnn_utils.pack_sequence(
        [t["label_traj"] for t in batch], enforce_sorted=False
    )
    return {
        "traj": packed_traj_batch,
        "label": torch.tensor([t["label"] for t in batch], dtype=torch.int64),
        "label_traj": packed_label_traj_batch,
        "mu": torch.tensor([t["mu"] for t in batch], dtype=torch.float32),
    }


class LoadPoincareTrajectoryDataset(Dataset):
    """Poincare Trajectory Dataset. This dataload unravels map datasets (collections of trajectories) and provides access to the individual trajectories in the Poincare section,
    augmented with a third coordinate which the is mass ratio parameter (to provide full specification of the system)

    """

    def __init__(self, root_dir, lookup_file, transform=None, coord_indices=None):
        """
        Args:
            root_dir (string): Path to find folders containing lookup files and data folders of .npz files
            lookup_file (string): name of the .csv file containing the lookup table of file names for the samples
            transform (callable, optional): Optional transform to be applied on a sample
        """

        self.lookup_df = pd.read_csv(root_dir + lookup_file, delimiter=",")
        print(len(self.lookup_df))
        self.lookup_df = self.lookup_df.where(path.isfile(self.data_dir + self.lookup_df['fname']))
        print(len(self.lookup_df))
        self.transform = transform
        self.coord_indices = coord_indices
        self.index_table = np.array(
            [
                (i, j)
                for i in range(self.lookup_df.shape[0])
                for j in range(self.lookup_df.iloc[i]["dim_pts"] ** 2)
            ]
        )
        self.length = self.index_table.shape[0]
        self.curr_npz_idx = None
        self.curr_npz = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        full_idx = self.index_table[idx]
        if self.curr_npz_idx != full_idx[0]:
            fname = self.lookup_df.iloc[full_idx[0]]["fname"]
            self.curr_npz = np.load(
                self.root_dir + f"npz_files/{fname}", allow_pickle=True
            )
            self.curr_npz_idx = full_idx[0]

        mu = self.lookup_df.iloc[full_idx[0]]["mu"]
        traj = self.curr_npz["coord_traj"][full_idx[1]][:, self.coord_indices]
        terminal_MEGNO_val = self.curr_npz["MEGNO_term"][full_idx[1]]
        crossing_times = self.curr_npz["crossing_times"][full_idx[1]]
        MEGNO_traj = self.curr_npz["MEGNO_traj"][full_idx[1]]
        sample = {
            "mu": mu,
            "traj": traj,
            "terminal_MEGNO_val": terminal_MEGNO_val,
            "crossing_times": crossing_times,
            "MEGNO_traj": MEGNO_traj,
        }

        if self.transform:
            sample = self.transform(sample)
        return sample


class LoadPoincareTrajectoryDatasetInMemory(object):
    """Poincare Trajectory Dataset. This dataload unravels map datasets (collections of trajectories) and provides access to the individual trajectories in the Poincare section,
    loading the entire dataset into a dictionary in memory
    """

    def __init__(
        self, lookup_dir, lookup_fname, data_dir, transform=None, coord_indices=None
    ):
        """
        Args:
            root_dir (string): Path to find folders containing lookup files and data folders of .npz files
            lookup_file (string): name of the .csv file containing the lookup table of file names for the samples
            transform (callable, optional): Optional transform to be applied on a sample
        """

        self.lookup_df = pd.read_csv(lookup_dir + lookup_fname, delimiter=",")
        self.data_dir = data_dir
        self.transform = transform
        self.coord_indices = coord_indices
        num_pts = self.lookup_df.iloc[0]["dim_pts"] ** 2
        L = len(self.lookup_df)
        self.entries = np.empty(L * num_pts, dtype=object)
        for i in range(L):
            print(f"{i:3d} / {len(self.lookup_df):3d}", end="\r")
            fname = self.lookup_df.iloc[i]["fname"]
            if path.isfile(self.data_dir + fname):  # check if file exists
                curr_npz = np.load(self.data_dir + fname, allow_pickle=True)
                for j in range(num_pts):
                    print(
                        f"{i:3d} / {len(self.lookup_df):3d} \t\t\t\t\t {j:3d} / {num_pts :3d}",
                        end="\r",
                    )
                    self.entries[num_pts * i + j] = {
                        "mu": self.lookup_df.iloc[i]["mu"],
                        "traj": curr_npz["coord_traj"][j][:, self.coord_indices],
                        "terminal_MEGNO_val": curr_npz["MEGNO_term"][j],
                        "crossing_times": curr_npz["crossing_times"][j],
                        "MEGNO_traj": curr_npz["MEGNO_traj"][j],
                    }
        self.length = len(self.entries)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.entries[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class LoadPoincareTrajectoryDatasetFromServer(Dataset):
    """Poincare Trajectory Dataset. This dataload unravels map datasets (collections of trajectories) and provides access to the individual trajectories in the Poincare section,
    augmented with a third coordinate which the is mass ratio parameter (to provide full specification of the system)

    Use something like
    'ssh -N -L localhost:8891:localhost:8003 electra'
    to configure port forwarding appropriately (above is for SSH access to data on bertha)
    in an open terminal (I don't use the -f flag so that this hack is easier to cleanup, rather than just leaving port forwarding running in the background until it's killed)

    """

    def __init__(
        self, root_dir, lookup_file, dim_pts=None, transform=None, coord_indices=None
    ):
        """
        Args:
            root_dir (string): Path to find folders containing lookup files and data folders of .npz files
            lookup_file (string): name of the .csv file containing the lookup table of file names for the samples
            transform (callable, optional): Optional transform to be applied on a sample
        """

        self.lookup_df = pd.read_csv(root_dir + lookup_file, delimiter=",")
        self.root_dir = root_dir
        self.transform = transform
        self.coord_indices = coord_indices
        self.index_table = np.array(
            [
                (i, j)
                for i in range(self.lookup_df.shape[0])
                for j in range(self.lookup_df.iloc[i]["dim_pts"] ** 2)
            ]
        )
        self.length = self.index_table.shape[0]
        self.curr_npz_idx = None
        self.curr_npz = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        full_idx = self.index_table[idx]
        if self.curr_npz_idx != full_idx[0]:
            fname = self.lookup_df.iloc[full_idx[0]]["fname"]
            if not path.isfile(
                self.root_dir + f"npz_files/{fname}"
            ):  # check if file needs to be downloaded
                with createSSHClient(
                    "localhost", 8891, "connorsteph", password=""
                ) as ssh:  # this only works when localhost port 8891 is forwarded to 8003 on electra
                    with SCPClient(ssh.get_transport()) as scp:
                        scp.get(
                            f"/home/connorsteph/SAM_traj_classification_data/npz_files/{fname}",
                            self.root_dir + "npz_files/",
                        )
            self.curr_npz = np.load(
                self.root_dir + f"npz_files/{fname}", allow_pickle=True
            )
            self.curr_npz_idx = full_idx[0]

        mu = self.lookup_df.iloc[full_idx[0]]["mu"]
        traj = self.curr_npz["coord_traj"][full_idx[1]][:, self.coord_indices]
        terminal_MEGNO_val = self.curr_npz["MEGNO_term"][full_idx[1]]
        crossing_times = self.curr_npz["crossing_times"][full_idx[1]]
        MEGNO_traj = self.curr_npz["MEGNO_traj"][full_idx[1]]
        sample = {
            "mu": mu,
            "traj": traj,
            "terminal_MEGNO_val": terminal_MEGNO_val,
            "crossing_times": crossing_times,
            "MEGNO_traj": MEGNO_traj,
        }

        if self.transform:
            sample = self.transform(sample)
        return sample


class LoadPoincareTrajectoryDatasetFromServerInMemory(object):
    """Poincare Trajectory Dataset. This dataload unravels map datasets (collections of trajectories) and provides access to the individual trajectories in the Poincare section,
    loading the entire dataset into a dictionary in memory

    """

    def __init__(
        self, root_dir, lookup_file, dim_pts=None, transform=None, coord_indices=None
    ):
        """
        Args:
            root_dir (string): Path to find folders containing lookup files and data folders of .npz files
            lookup_file (string): name of the .csv file containing the lookup table of file names for the samples
            transform (callable, optional): Optional transform to be applied on a sample
        """

        self.lookup_df = pd.read_csv(root_dir + lookup_file, delimiter=",")
        self.root_dir = root_dir
        self.transform = transform
        self.coord_indices = coord_indices
        self.entries = []
        for i in range(len(self.lookup_df)):
            print(f"{i:3d} / {len(self.lookup_df):3d}", end="\r")
            fname = self.lookup_df.iloc[i]["fname"]
            if not path.isfile(
                self.root_dir + f"npz_files/{fname}"
            ):  # check if file needs to be downloaded
                with createSSHClient(
                    "localhost", 8891, "connorsteph", password=""
                ) as ssh:  # this only works when localhost port 8891 is forwarded to 8003 on electra
                    with SCPClient(ssh.get_transport()) as scp:
                        scp.get(
                            f"/home/connorsteph/SAM_traj_classification_data/npz_files/{fname}",
                            self.root_dir + "npz_files/",
                        )
            curr_npz = np.load(self.root_dir + f"npz_files/{fname}", allow_pickle=True)
            for j in range(self.lookup_df.iloc[i]["dim_pts"] ** 2):
                mu = self.lookup_df.iloc[i]["mu"]
                traj = curr_npz["coord_traj"][j][:, self.coord_indices]
                terminal_MEGNO_val = curr_npz["MEGNO_term"][j]
                crossing_times = curr_npz["crossing_times"][j]
                MEGNO_traj = curr_npz["MEGNO_traj"][j]
                self.entries.append(
                    {
                        "mu": mu,
                        "traj": traj,
                        "terminal_MEGNO_val": terminal_MEGNO_val,
                        "crossing_times": crossing_times,
                        "MEGNO_traj": MEGNO_traj,
                    }
                )
        self.entries = np.array(self.entries, dtype=object)
        np.random.shuffle(self.entries)
        self.length = len(self.entries)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.entries[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class LoadPoincareMapGridDataset(Dataset):
    def __init__(self, lookup_dir, lookup_fname, data_dir, transform=None, coord_indices=None
    ):
        """
        Args:
            root_dir (string): Path to find folders containing lookup files and data folders of .npz files
            lookup_file (string): name of the .csv file containing the lookup table of file names for the samples
            transform (callable, optional): Optional transform to be applied on a sample
        """
        
        self.lookup_df = pd.read_csv(lookup_dir + lookup_fname, delimiter=",")
        is_file = lambda x: path.isfile(data_dir + x)
        is_file = np.vectorize(is_file)
        # print(len(self.lookup_df))
        self.lookup_df = self.lookup_df[is_file(self.lookup_df['fname'])]
        # print(len(self.lookup_df))
        self.data_dir = data_dir
        self.transform = transform
        self.coord_indices = coord_indices
        self.length = len(self.lookup_df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fname = self.lookup_df.iloc[idx]["fname"]
        curr_npz = np.load(self.data_dir + fname, allow_pickle=True)
        mu = self.lookup_df.iloc[idx]["mu"]
        traj = curr_npz["coord_traj"]
        terminal_MEGNO_val = curr_npz["MEGNO_term"]
        crossing_times = curr_npz["crossing_times"]
        MEGNO_traj = curr_npz["MEGNO_traj"]
        sample = {
            "mu": mu,
            "map_list": traj,
            "terminal_MEGNO_val_list": terminal_MEGNO_val,
            "crossing_times_list": crossing_times,
            "MEGNO_traj_list": MEGNO_traj,
        }
        if self.transform:
            return self.transform(sample)
        else:
            return sample

class LoadPoincareMapGridDatasetInMemory(Dataset):
    def __init__(
        self, lookup_dir, lookup_fname, data_dir, transform=None, coord_indices=None
    ):
        """
        Args:
            root_dir (string): Path to find folders containing lookup files and data folders of .npz files
            lookup_file (string): name of the .csv file containing the lookup table of file names for the samples
            transform (callable, optional): Optional transform to be applied on a sample
        """
        # sanitize lookup_df
        self.lookup_df = pd.read_csv(lookup_dir + lookup_fname, delimiter=",")
        is_file = lambda x: path.isfile(data_dir + x)
        is_file = np.vectorize(is_file)
        # print(len(self.lookup_df))
        self.lookup_df = self.lookup_df[is_file(self.lookup_df['fname'])]
        self.data_dir = data_dir
        self.transform = transform
        self.coord_indices = coord_indices
        self.entries = np.empty(len(self.lookup_df), dtype=object)


        for i in range(len(self.lookup_df)):
            print(f"{i:3d} / {len(self.lookup_df):3d}", end="\r")
            fname = self.lookup_df.iloc[i]["fname"]
            if path.isfile(
                self.data_dir + fname
            ):  # check if file needs to be downloaded
                curr_npz = np.load(self.data_dir + fname, allow_pickle=True)
                mu = self.lookup_df.iloc[i]["mu"]
                traj = curr_npz["coord_traj"]
                terminal_MEGNO_val = curr_npz["MEGNO_term"]
                crossing_times = curr_npz["crossing_times"]
                MEGNO_traj = curr_npz["MEGNO_traj"]
                sample = {
                    "mu": mu,
                    "map_list": traj,
                    "terminal_MEGNO_val_list": terminal_MEGNO_val,
                    "crossing_times_list": crossing_times,
                    "MEGNO_traj_list": MEGNO_traj,
                }
                if self.transform:
                    sample = self.transform(sample)
                self.entries[i] = sample

        # self.entries = np.array(self.entries, dtype=object)
        self.length = len(self.entries)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.entries[idx]


class LoadPoincareMapGridDatasetFromServer(Dataset):
    def __init__(
        self,
        root_dir,
        lookup_file,
        remote_dir,
        dim_pts=None,
        transform=None,
        coord_indices=None,
    ):
        """
        Args:
            root_dir (string): Path to find folders containing lookup files and data folders of .npz files
            lookup_file (string): name of the .csv file containing the lookup table of file names for the samples
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.lookup_df = pd.read_csv(root_dir + lookup_file, delimiter=",")
        self.remote_dir = remote_dir
        self.length = len(self.lookup_df)
        self.root_dir = root_dir
        self.transform = transform
        self.coord_indices = coord_indices
        for i in range(len(self.lookup_df)):
            print(f"{i:3d} / {len(self.lookup_df):3d}", end="\r")
            fname = self.lookup_df.iloc[i]["fname"]
            if not path.isfile(
                self.root_dir + f"npz_files/{fname}"
            ):  # check if file needs to be downloaded
                with createSSHClient(
                    "localhost", 8891, "connorsteph", password=""
                ) as ssh:  # this only works when localhost port 8891 is forwarded to 8003 (bertha) on electra
                    with SCPClient(ssh.get_transport()) as scp:
                        scp.get(
                            f"{self.remote_dir}{fname}",
                            self.root_dir + "npz_files/",
                        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fname = self.lookup_df.iloc[idx]["fname"]
        if not path.isfile(
            self.root_dir + f"npz_files/{fname}"
        ):  # check if file needs to be downloaded
            with createSSHClient(
                "localhost", 8891, "connorsteph", password=""
            ) as ssh:  # this only works when localhost port 8891 is forwarded to 8003 on electra
                with SCPClient(ssh.get_transport()) as scp:
                    scp.get(
                        f"{self.remote_dir}{fname}",
                        self.root_dir + "npz_files/",
                    )
        curr_npz = np.load(self.root_dir + f"npz_files/{fname}", allow_pickle=True)
        mu = self.lookup_df.iloc[idx]["mu"]
        traj = curr_npz["coord_traj"]
        terminal_MEGNO_val = curr_npz["MEGNO_term"]
        crossing_times = curr_npz["crossing_times"]
        MEGNO_traj = curr_npz["MEGNO_traj"]
        sample = {
            "mu": mu,
            "map_list": traj,
            "terminal_MEGNO_val_list": terminal_MEGNO_val,
            "crossing_times_list": crossing_times,
            "MEGNO_traj_list": MEGNO_traj,
        }
        if self.transform:
            return self.transform(sample)
        else:
            return sample


class LoadPoincareMapGridDatasetFromServerInMemory(Dataset):
    def __init__(
        self, root_dir, lookup_file, dim_pts=None, transform=None, coord_indices=None
    ):
        """
        Args:
            root_dir (string): Path to find folders containing lookup files and data folders of .npz files
            lookup_file (string): name of the .csv file containing the lookup table of file names for the samples
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.lookup_df = pd.read_csv(root_dir + lookup_file, delimiter=",")
        self.root_dir = root_dir
        self.transform = transform
        self.coord_indices = coord_indices
        self.entries = []
        for i in range(len(self.lookup_df)):
            print(f"{i:3d} / {len(self.lookup_df):3d}", end="\r")
            fname = self.lookup_df.iloc[i]["fname"]
            if not path.isfile(
                self.root_dir + f"npz_files/{fname}"
            ):  # check if file needs to be downloaded
                with createSSHClient(
                    "localhost", 8891, "connorsteph", password=""
                ) as ssh:  # this only works when localhost port 8891 is forwarded to 8003 (bertha) on electra
                    with SCPClient(ssh.get_transport()) as scp:
                        scp.get(
                            f"/home/connorsteph/SAM_traj_classification_data/npz_files/{fname}",
                            self.root_dir + "npz_files/",
                        )
            curr_npz = np.load(self.root_dir + f"npz_files/{fname}", allow_pickle=True)
            mu = self.lookup_df.iloc[i]["mu"]
            traj = curr_npz["coord_traj"]
            terminal_MEGNO_val = curr_npz["MEGNO_term"]
            crossing_times = curr_npz["crossing_times"]
            MEGNO_traj = curr_npz["MEGNO_traj"]
            sample = {
                "mu": mu,
                "map_list": traj,
                "terminal_MEGNO_val_list": terminal_MEGNO_val,
                "crossing_times_list": crossing_times,
                "MEGNO_traj_list": MEGNO_traj,
            }
            if self.transform:
                sample = self.transform(sample)
            self.entries.append(sample)

        self.entries = np.array(self.entries, dtype=object)
        self.length = len(self.entries)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.entries[idx]


class FineThresholdMEGNOValues(object):
    """Discretize terminal MEGNO values to class labels"""

    def __init__(self, breakpts=[10.0, 3.0, 2.0]):
        self.breakpts = breakpts

    def threshold(self, x):
        if x > self.breakpts[0]:  # chaotic
            return 3
        elif x > self.breakpts[1]:  # unclear
            return 2
        elif x > self.breakpts[2]:  # near-unstable quasiperiodic
            return 1
        else:  # stable
            return 0

    def threshold_traj(self, x):
        y = 0 * np.ones(x.shape)
        y[x >= self.breakpts[2]] = 1
        y[x >= self.breakpts[1]] = 2
        y[x >= self.breakpts[0]] = 3

        return y

    def __call__(self, sample):
        return {
            "traj": sample["traj"],
            "label": self.threshold(sample["terminal_MEGNO_val"]),
            "label_traj": self.threshold_traj(sample["MEGNO_traj"]),
            "mu": sample["mu"],
            "crossing_times": sample["crossing_times"],
        }


class GridFineThresholdMEGNOValues(object):
    """Discretize terminal MEGNO values to class labels"""

    def __init__(self, breakpts=[10.0, 3.0, 2.0]):
        self.breakpts = breakpts

    def threshold(self, x):
        y = np.zeros(x.shape, np.int64)
        y[x >= self.breakpts[2]] = 1
        y[x >= self.breakpts[1]] = 2
        y[x >= self.breakpts[0]] = 3
        return y

    def vector_threshold(self, x):
        # threshold each label trajectory
        for idx, item in enumerate(x):
            y = np.zeros(item.shape, np.int64)
            y[item >= self.breakpts[2]] = 1
            y[item >= self.breakpts[1]] = 2
            y[item >= self.breakpts[0]] = 3
            x[idx] = y
        return x

    def __call__(self, sample):
        term_label_list = self.threshold(sample["terminal_MEGNO_val_list"])
        transient_label_list = self.vector_threshold(sample["MEGNO_traj_list"])
        return {
            "map_list": sample["map_list"],
            "crossing_times_list": sample["crossing_times_list"],
            "transient_label_list": transient_label_list,
            "label_list": term_label_list,
            "mu": sample["mu"],
        }


class ThresholdMEGNOValues(object):
    """Discretize terminal MEGNO values to class labels"""

    def __init__(self, breakpts=[10.0, 3.0, 2.0]):
        self.breakpts = breakpts

    def threshold(self, x):
        if x > self.breakpts[0]:  # chaotic
            return 1
        elif x > self.breakpts[1]:  # unclear
            return 0
        elif x > self.breakpts[2]:  # near-unstable quasiperiodic
            return 0
        else:  # stable
            return 0

    def threshold_traj(self, x):
        y = 0 * np.ones(x.shape)
        y[x >= self.breakpts[0]] = 1
        return y

    def __call__(self, sample):
        return {
            "traj": sample["traj"],
            "label": self.threshold(sample["terminal_MEGNO_val"]),
            "label_traj": self.threshold_traj(sample["MEGNO_traj"]),
            "mu": sample["mu"],
            "crossing_times": sample["crossing_times"],
        }


class TimeCrop(object):
    """Finds index of last crossing time less than time t, and returns sample with trajectories cropped to this index"""

    def __init__(self, t):
        self.t = t

    def __call__(self, sample):
        crossing_times = sample["crossing_times"]
        if np.max(crossing_times) < self.t:
            return {
                "traj": sample["traj"],
                "MEGNO_traj": sample["MEGNO_traj"],
                "terminal_MEGNO_val": sample["terminal_MEGNO_val"],
                "mu": sample["mu"],
                "crossing_times": sample["crossing_times"],
            }
        else:
            idx = np.argmin(crossing_times < self.t) - 1
            return {
                "traj": sample["traj"][:idx],
                "MEGNO_traj": sample["MEGNO_traj"],
                "terminal_MEGNO_val": sample["terminal_MEGNO_val"],
                "mu": sample["mu"],
                "crossing_times": sample["crossing_times"],
            }


class ToTensor(object):
    """Convert ndarrays in sample to Tensors. (with dimension for expected colour channel)"""

    def __call__(self, sample):
        return {
            "traj": torch.tensor(sample["traj"], dtype=torch.float),
            "label": torch.tensor(sample["label"], dtype=torch.int64),
            "label_traj": torch.tensor(sample["label_traj"], dtype=torch.int),
            "crossing_times": torch.tensor(sample["crossing_times"], dtype=torch.float),
            "mu": torch.tensor(np.array([sample["mu"]]), dtype=torch.float),
        }


# class LoadPoincareMapGridDataset(Dataset):
#     """Poincare Map Dataset."""

#     def __init__(self, root_dir, lookup_file, transform=None):
#         """
#         Args:
#             root_dir (string): Path to find folders containing lookup files and data folders of .npz files
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.lookup_df = pd.read_csv(root_dir + lookup_file, delimiter=",")
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return self.lookup_df.shape[0]

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         fname = self.lookup_df.iloc[idx]["fname"]
#         npz_file = np.load(self.root_dir + f"npz_files/{fname}", allow_pickle=True)
#         map_list, transient_MEGNO_val_list, terminal_MEGNO_val_list = (
#             npz_file["coord_traj"],
#             npz_file["MEGNO_traj"],
#             npz_file["MEGNO_term"],
#         )
#         transient_MEGNO_val_list = np.array(
#             [x[-1] for x in transient_MEGNO_val_list], dtype=np.float64
#         )
#         sample = {
#             "map_list": map_list,
#             "transient_MEGNO_val_list": transient_MEGNO_val_list,
#             "terminal_MEGNO_val_list": terminal_MEGNO_val_list,
#             "mu": self.lookup_df.iloc[idx]["mu"],
#         }

#         if self.transform:
#             sample = self.transform(sample)
#         return sample


class GridThresholdMEGNOValues(object):
    """Discretize terminal MEGNO values to class labels"""

    def __init__(self, breakpt=10.0):
        self.breakpt = breakpt

    def threshold(self, x):
        y = np.zeros(x.shape, np.int64)
        y[x >= self.breakpt] = 1
        return y

    def vector_threshold(self, x):
        # threshold each label trajectory
        for idx, label_traj in enumerate(x):
            y = np.zeros(label_traj.shape, np.int64)
            y[label_traj >= self.breakpt] = 1
            x[idx] = y
        return x

    def __call__(self, sample):
        term_label_list = self.threshold(sample["terminal_MEGNO_val_list"])
        transient_label_list = self.vector_threshold(sample["MEGNO_traj_list"])
        return {
            "map_list": sample["map_list"],
            "crossing_times_list": sample["crossing_times_list"],
            "transient_label_list": transient_label_list,
            "label_list": term_label_list,
            "mu": sample["mu"],
        }


class MapListtoGrid:
    """
    Takes a list of dim_pts**2 sequences of coordinates to a nd-array of size (dim_pts, dim_pts, min_traj_length)
    and makes the same conversion for the respective label per grid position
    """

    def __init__(self, coords=None, crop_to_min=False, max_length=None):
        self.crop = crop_to_min
        self.coords = coords
        self.max_length = max_length
        self.pad_to = None
        self.crop_to = None

    def pad_to_length(self, input):
        L = len(input)
        pad_len = self.pad_to - L
        if pad_len > 0:
            self.extra_padding_args = [(0, 0)] * (input.ndim - 1)
            input = np.pad(
                input,
                [(0, pad_len)] + self.extra_padding_args,
                mode="constant",
                constant_values=0,
            )
        return input

    def crop_to_length(self, input):
        L = len(input)
        if L > self.crop_to:
            return input[: self.crop_to]
        else:
            return input

    def flat_to_2d_array(self, flat_array, dtype=None):
        if dtype is None:
            dtype = flat_array[0].dtype
        dim_pts = int(np.sqrt(len(flat_array)))
        assert dim_pts**2 == len(
            flat_array
        )  # make sure the original array is a flattened square
        output = flat_array
        if isinstance(flat_array[0], np.ndarray):  # check if output array should be 3-D
            output = np.array(
                [self.pad_to_length(item) for item in flat_array], dtype=object
            )
            output = np.array([self.crop_to_length(item) for item in output])
        return np.array(
            [[output[j + i * dim_pts] for j in range(dim_pts)] for i in range(dim_pts)],
            dtype=dtype,
        )

    def __call__(self, sample):
        map_list = sample["map_list"]
        if self.coords:
            map_list = np.array([map[:, self.coords] for map in map_list], dtype=object)
        label_list = sample["label_list"]
        transient_label_list = sample["transient_label_list"]
        crossing_time_list = sample["crossing_times_list"]

        if self.crop:
            assert (
                self.max_length is None
            )  # specifying a max length is incompatible with cropping to min
            self.crop_to = np.min([len(map) for map in map_list])
            self.pad_to = self.crop_to
        else:
            if self.max_length is None:
                # no max length specified -- use the length of the longest trajectory
                self.pad_to = np.max([len(map) for map in map_list])
                self.crop_to = self.pad_to
            else:
                self.pad_to = self.max_length
                self.crop_to = self.pad_to

        grid_of_trajectories = self.flat_to_2d_array(map_list, dtype=np.float32)
        grid_of_transient_labels = self.flat_to_2d_array(
            transient_label_list, dtype=np.float32
        )
        grid_of_crossing_times = self.flat_to_2d_array(
            crossing_time_list, dtype=np.float32
        )
        grid_of_labels = self.flat_to_2d_array(label_list, dtype=np.int32)
        return {
            "map_grid": grid_of_trajectories,
            "transient_label_grid": grid_of_transient_labels,
            "crossing_time_grid": grid_of_crossing_times,
            "label_grid": grid_of_labels,
            "mu": sample["mu"],
        }


class GridToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return {
            "map_grid": torch.tensor(sample["map_grid"], dtype=torch.float),
            "transient_label_grid": torch.tensor(
                sample["transient_label_grid"], dtype=torch.long
            ),
            "crossing_time_grid": torch.tensor(
                sample["crossing_time_grid"], dtype=torch.float
            ),
            "label_grid": torch.tensor(sample["label_grid"], dtype=torch.long),
            "mu": torch.tensor(sample["mu"], dtype=torch.float),
        }


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
        vmin=0,
        vmax=255,
        E=1.0,
        alpha=None,
        coords=None,
        normalized=False,
        n_classes=None,
        x_range=None,
        y_range=None,
    ):
        self.img_width = img_width
        self.vmin = vmin
        self.vmax = vmax
        self.alpha = alpha
        self.E = E
        self.coords = coords
        self.normalized = normalized
        self.n_classes = n_classes
        self.x_range=x_range
        self.y_range=y_range

    def __call__(self, sample):
        map_list = sample["map_list"]
        if self.n_classes:
            label_list = sample["label_list"]
        mu = sample["mu"]
        # map_list = np.concatenate(map_list, axis=0)[:, self.coords]
        # label_list = np.concatenate(label_list, axis=0)
        if self.n_classes:
            pixel_class_masks = -np.ones(
                (self.n_classes, self.img_width, self.img_width), dtype=np.int8
            )
            # pixel_class_masks[0, ...] = -1
        pixel_array = self.vmax * np.ones(
            (self.img_width, self.img_width), dtype=np.uint8
        )
        map_list = [map[:, self.coords] for map in map_list]
        if self.x_range is None: 
            x_range = [mu-1-.25, mu+.25]
        else:
            x_range = self.x_range
        if self.y_range is None:
            y_range = [
                np.min([np.min(map[:, 1]) for map in map_list]),
                np.max([np.max(map[:, 1]) for map in map_list]),
            ]
        else:
            y_range = self.y_range
        for idx, map in enumerate(map_list):
            if self.normalized:
                # warp Poincare map to fill unit square
                x_scaled = map[:, 0] * (mu - 1.0) / self.E  # warp x_vals to (0,1) range
                y_scaled = (
                    map[:, 1]
                    / (2 * np.sqrt(2 * (mu + 1) * (self.E - (mu - 1) * map[:, 0])))
                    + 0.5
                )  # warp y vals to (0,1) range
            else:
                # remove all items in map with coord 0 outside of x_range and coord 1 outside of y_range
                indices = np.greater_equal(map[:,0], x_range[0]) \
                * np.less_equal(map[:,0], x_range[1]) \
                * np.greater_equal(map[:,1], y_range[0]) \
                * np.less_equal(map[:,1], y_range[1])
                map = map[indices, :]
                # scale Poincare map to fit into the unit square
                x_scaled = (map[:, 0] - x_range[0]) / (
                    x_range[1] - x_range[0]
                )  # scale x_vals to (0,1) range
                y_scaled = (map[:, 1] - y_range[0]) / (
                    y_range[1] - y_range[0]
                )  # scale y_vals to (0,1) range

            # discretize crossings to grid coordinates
            x_scaled = np.uint8(np.round(float(self.img_width - 1) * x_scaled))
            y_scaled = np.uint8(np.round(float(self.img_width - 1) * (1.0 - y_scaled)))
            pixel_array[y_scaled, x_scaled] = np.uint8(
                self.alpha * pixel_array[y_scaled, x_scaled]
            )  # shade pixels with alpha by counting crossings per square
            if self.n_classes:  # set pixel class mask
                pixel_class_masks[int(label_list[idx]), y_scaled, x_scaled] = int(
                    label_list[idx]
                )
        # normalize pixel values from [vmin, vmax] range to [-1,1] range

        image = 2.0 * (
            (pixel_array.astype(np.float16) - self.vmin) / float(self.vmax - self.vmin)
        )
        image -= 1.0
        if self.n_classes:
            pixel_classes = np.max(
                pixel_class_masks, axis=0
            )  # get the max class for each pixel
            pixel_classes += 1  # shift class labels to make 0 the background class
            return {"image": image, "label_image": pixel_classes, "mu": sample["mu"]}
        else:
            return {"image": image, "mu": sample["mu"]}


class RandomCroppedMapListToPixelArray(object):
    """Converts a numpy array of Poincare section crossing coordinates to an image array of specified width,
    after making a transformation of the map boundary to a square, using a transformation specific to the SAM with known mu and E

    Args:
        img_width (int): Desired width of image
        alpha (float, optional): alpha value for pixel darkening per hit
    """

    def __init__(
        self,
        img_width,
        vmin=0,
        vmax=255,
        E=1.0,
        alpha=None,
        min_length=1,
        coords=None,
        normalized=False,
        x_range=None,
        y_range=None,
    ):
        self.img_width = img_width
        self.vmin = vmin
        self.vmax = vmax
        self.alpha = alpha
        self.E = E
        self.min_length = min_length
        self.coords = coords
        self.normalized = normalized
        self.x_range=x_range
        self.y_range=y_range

    def __call__(self, sample):
        map_list = sample["map_list"]
        mu = sample["mu"]
        if self.y_range is None:
            y_range = [
                np.min([np.min(map[:, 1]) for map in map_list]),
                np.max([np.max(map[:, 1]) for map in map_list]),
            ]
        else:
            y_range = self.y_range
        if self.x_range is None: # this is only for CR3BP
            x_range = [mu-1-.25, mu+.25]
            # x_range = [
            #     np.min([np.min(map[:, 0]) for map in map_list]),
            #     np.max([np.max(map[:, 0]) for map in map_list]),
            # ]
        else:
            x_range = self.x_range

        if self.min_length is not None:
            map_list = np.random.permutation(map_list)[
                : np.random.randint(self.min_length, len(map_list))
            ]  # select a random subset of trajectories
        sample = {"mu": mu, "map_list": map_list}
        pixel_array = self.vmax * np.ones(
            (self.img_width, self.img_width), dtype=np.uint8
        )
        map_list = [map[:, self.coords] for map in map_list]
        
        for map in map_list:
            if self.normalized:
                # warp Poincare map to fill unit square
                x_scaled = map[:, 0] * (mu - 1.0) / self.E  # warp x_vals to (0,1) range
                y_scaled = (
                    map[:, 1]
                    / (2 * np.sqrt(2 * (mu + 1) * (self.E - (mu - 1) * map[:, 0])))
                    + 0.5
                )  # warp y vals to (0,1) range
            else:
                # remove all items in map with coord 0 outside of x_range and coord 1 outside of y_range
                indices = np.greater_equal(map[:,0], x_range[0]) \
                    * np.less_equal(map[:,0], x_range[1]) \
                    * np.greater_equal(map[:,1], y_range[0]) \
                    * np.less_equal(map[:,1], y_range[1])
                map = map[indices, :]
                # scale Poincare map to fit into the unit square
                x_scaled = (map[:, 0] - x_range[0]) / (
                    x_range[1] - x_range[0]
                )  # scale x_vals to (0,1) range
                y_scaled = (map[:, 1] - y_range[0]) / (
                    y_range[1] - y_range[0]
                )  # scale y_vals to (0,1) range

            # discretize crossings to grid coordinates
            x_scaled = np.uint8(np.round(float(self.img_width - 1) * x_scaled))
            y_scaled = np.uint8(np.round(float(self.img_width - 1) * (1.0 - y_scaled)))
            pixel_array[y_scaled, x_scaled] = np.uint8(
                self.alpha * pixel_array[y_scaled, x_scaled]
            )  # shade pixels with alpha by counting crossings per square
        # normalize pixel values from [vmin, vmax] range to [-1,1] range
        image = 2.0 * (
            (pixel_array.astype(np.float16) - self.vmin) / float(self.vmax - self.vmin)
        )
        image -= 1.0
        return {"image": image, "mu": sample["mu"]}
