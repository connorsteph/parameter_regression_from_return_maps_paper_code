import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm

import multiprocessing as mp


def worker_function(items):
    val_list, pid = items
    f_names = []
    if os.getpid() == pid:
        print("Doing stuff!")
    for val in val_list:
        f_names.append(f"test_{val}.npy")
        # np.save(f_names[-1], np.random.rand(100, 100))
    return f_names


if __name__ == "__main__":
    # num_processes = 8
    # p = mp.Pool(processes=num_processes)
    # # select one of the processes to print out from
    # print(p._pool[0].pid)

    # all_vals = np.arange(40)
    # for item in np.array_split(all_vals, num_processes):
    #     break
    # f_names = p.map_async(worker_function, zip(np.array_split(all_vals, num_processes), p._pool[0].pid * np.ones(num_processes))).get()
    # p.close()
    # print(f_names)

    def track_job(job, num_tasks, update_interval=.1):
        with tqdm(total=num_tasks) as pbar:
            num_left = job._number_left
            while job._number_left > 0:
                time.sleep(update_interval)
                pbar.update(num_left - job._number_left)
                num_left = job._number_left

        return job.get()



    def hi(x): #This must be defined before `p` if we are to use in the interpreter
        time.sleep(x//20)
        return x

    a = [x for x in range(50)]
    num_tasks = len(a)

    with mp.Pool() as p:
        res = p.map_async(hi,a)
        results = track_job(res, num_tasks)
        print(results)
    
