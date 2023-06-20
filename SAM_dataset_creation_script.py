from HamiltonianSystemsTools.utils import HamiltonianSystem

import heyoka as hy
import numpy as np
from hashlib import sha256
from datetime import datetime
import configparser
import argparse
import yaml


from pathlib import Path
from tqdm import tqdm

"""
    After running this script your directory structure should look like 

    root_dir/
    ├─  lookup_file.csv
    ├─  logfile_{datetime.now()}.txt
    ├─  npz_files/
    |   ├─  {random_filename_0}.npz
    |   ├─  {random_filename_1}.npz
    |   ├─  ...
"""

"""
Define the Swinging Atwood's Machine in terms of its Lagrangian,
written in Lagrange coordinates (r, phi, r_dot, phi_dot)
"""


def T_func(q, qd, params):
    return 1 / 2 * params[0] * (qd[0] ** 2) + 1 / 2 * (
        (qd[0] ** 2) + (q[0] ** 2) * (qd[1] ** 2)
    )  # SAM kinetic term


def U_func(q, qd, params):
    return params[0] * q[0] * params[1] - q[0] * params[1] * hy.cos(
        q[1]
    )  # SAM potential term


def L_func(q=None, qd=None, params=None):
    return T_func(q=q, qd=qd, params=params) - U_func(q=q, qd=qd, params=params)


SAM = HamiltonianSystem(num_coords=2, num_params=2, L_func=L_func)

# define Heyoka events and callbacks to record points from the Poincaré map


class ps_cb_time_stamp_cos_batch:
    def __init__(self, batch_size=4):
        self.crossing_coords = list([] for _ in range(batch_size))

    def __call__(self, ta, t, d_sgn, b_idx):
        ta.update_d_output(t)
        if np.cos(ta.d_output[1, b_idx]) > 0:
            self.crossing_coords[b_idx].append(
                np.array([*ta.d_output.copy()[:, b_idx], t])
            )
        return False


SAM.ps_cb_batch = ps_cb_time_stamp_cos_batch()

SAM_ps_ev_batch = hy.nt_event_batch(
    hy.sin(SAM.q[1]), direction=hy.event_direction.positive, callback=SAM.ps_cb_batch
)

# # we're gonna run into some NaN's when we're setting ICs :)
# np.seterr(all="ignore")

# # choice of system parameters for the SAM other than the mass ratio
# g = 1.0
# E = 1.0

# # input parameters of the integration method
# t_lim = 1e3  # how long to propagate system while recording section crossings
# """
# when label_t_lim > t_lim, the system will be propagated all the way out to label_t_lim WITHOUT recording section crossings past t_lim, but while maintaining the MEGNO calculation.
# This allows us to only record the Poincare section crossings that occur before t_lim, but still have the ability to look at the value of the MEGNO indicator after an
# arbitrarily long time lim (here at label_t_lim)

# i.e. label_t_lim only affects the value of 'terminal_label' in the dataset, which correspond to the value of the MEGNO indicator after propagation until label_t_lim
# """

# # size of the grid of points to be sampled along each dimension
# dim_pts = 3
# # mu_interval = 0.25
# mu_interval = None
# mu_pts = 100
# mu_min = 1.5
# mu_max = 15.0
# # mu_list = np.arange(mu_min, mu_max + mu_interval, mu_interval)
# mu_list = np.linspace(mu_min, mu_max, mu_pts)

# # integrator parameters
# batch_size = 4  # number of batches to use for the batch integrator -- 4 is the correct number in 99.5% of cases. See the Heyoka batch documentation for more info.
# max_workers = (
#     6  # number of cores to use -- scaling this up e.g. on a 256 core server CPU would be a good idea for large (or expensive) datasets
# )

# # num_samples = samples_per_mu * len(mu_list)
# num_samples = len(mu_list)

# # write to log file
# datetime_now_str = str(datetime.now()).split(".")[0]
# datetime_now_str = (
#     datetime_now_str.replace(" ", "_").replace(":", "hr", 1).replace(":", "min", 1)
#     + "sec"
# )
# logfile_path = root_dir + f"log_{datetime_now_str}.txt"
# log_str = f"""Logfile for experiment running at {str(datetime.now())}

# Experiment settings:
# g = {g}
# E = {E}
# t_lim = {t_lim}
# dim_pts = {dim_pts}
# mu_min = {mu_min}
# mu_max = {mu_max}
# mu_interval = {mu_interval}
# mu_pts = {mu_pts}
# """


def main(args):
    np.random.seed(42)
    # np.seterr(all="ignore") # we're gonna run into some NaN's when we're setting initial states :)

    # parse config file in the same way as in the training script
    config = configparser.ConfigParser()
    config.read(args.cfg)
    # FILESYSTEM params
    # -------------------------------------------------------
    root_dir = config.get("FILESYSTEM", "root_dir")
    # -------------------------------------------------------
    # HARDWARE params
    # -------------------------------------------------------
    num_workers = config.getint("HARDWARE", "num_workers")
    batch_size = config.getint("HARDWARE", "batch_size")
    # -------------------------------------------------------
    # DATASET params
    # -------------------------------------------------------
    g = config.getfloat("DATASET", "g")
    E = config.getfloat("DATASET", "E")
    mu_min = config.getfloat("DATASET", "mu_min")
    mu_max = config.getfloat("DATASET", "mu_max")
    t_lim = config.getfloat("DATASET", "t_lim")
    dim_pts = config.getint("DATASET", "dim_pts")
    num_samples = config.getint("DATASET", "num_samples")
    # -------------------------------------------------------

    # the name of the lookup file to use or create
    lookup_file_path = root_dir + "main_lookup.csv"
    # path of the directory to store the actual datafiles in
    data_file_dir = root_dir + "npz_files/"
    Path(data_file_dir).mkdir(parents=True, exist_ok=True)

    # write config to log file
    datetime_now_str = str(datetime.now()).split(".")[0]
    datetime_now_str = (
        datetime_now_str.replace(" ", "_").replace(
            ":", "hr", 1).replace(":", "min", 1)
        + "sec"
    )
    logfile_path = root_dir + f"log_{datetime_now_str}.yaml"
    with open(logfile_path, "w+") as logfile:
        yaml.dump(
            {"g": g, "E": E, "t_lim": t_lim, "dim_pts": dim_pts,
                "mu_min": mu_min, "mu_max": mu_max, "num_samples": num_samples
             },
            logfile
        )

    header = "" if Path.exists(
        Path(lookup_file_path)) else "mu,t_lim,dim_pts,fname,logfile"

    ta = None
    order = len(SAM.get_ODE_sys())
    iter = 0

    mu_list = np.linspace(mu_min, mu_max, num_samples)

    ta_template = SAM.generate_poincare_points_ensemble_batch(
                section_event=SAM_ps_ev_batch,
                t_lim=t_lim,
                params=[0., 0.],
                integral_constraint=SAM.H,
                integral_value=E,
                constrained_idx=3,
                init_state_list=initial_state_list,
                max_workers=num_workers,
                batch_size=batch_size,
                ta=ta,
                output_ta_template=True,
            )

    with tqdm(total=len(mu_list)) as pbar:
        for mu in tqdm((mu_list)):
            initial_state_list = (
                [r, 0.0, p_r, None]
                for r in np.linspace(1e-3, E / (mu - 1.0) - 1e-3, dim_pts)
                for p_r in np.linspace(
                    -np.sqrt(2 * (mu + 1.0) * (E - (mu - 1.0) * r)) + 1e-3,
                    np.sqrt(2 * (mu + 1.0) * (E - (mu - 1.0) * r)) - 1e-3,
                    dim_pts,
                )
            )
            initial_state_list = SAM.fill_initial_state_list(
                initial_state_list, SAM.H, E, 3, [mu, g]
            )

            map_list = SAM.generate_poincare_points_ensemble_batch(
                section_event=SAM_ps_ev_batch,
                t_lim=t_lim,
                params=[mu, g],
                integral_constraint=SAM.H,
                integral_value=E,
                constrained_idx=3,
                init_state_list=initial_state_list,
                max_workers=num_workers,
                batch_size=batch_size,
                ta=ta_template,
            )

            coord_trajectories = np.empty(dim_pts**2, object)

            coord_trajectories[:] = [map[:, :order] for map in map_list]

            hash_str = str([mu, g, E, t_lim]) + str(initial_state_list)
            fname = str(
                sha256(hash_str.encode()).hexdigest()
            )  # generate a unique filename string by hashing initial conditions + system parameters
            fname = fname + ".npz"

            # save the data to a .npz file
            np.savez_compressed(
                data_file_dir + fname,
                coord_traj=coord_trajectories,
            )

            # append the new file, along with it's parameters, and the corresponding logfile to the lookup table
            row = [[mu, t_lim, dim_pts, fname, f"log_{datetime_now_str}.txt"]]
            with open(lookup_file_path, "ab") as lookup_file:
                np.savetxt(
                    lookup_file, row, fmt="%s", delimiter=",", header=header, comments=""
                )
            header = ""

            iter += 1
            pbar.update(1)
        pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    main(parser.parse_args())
