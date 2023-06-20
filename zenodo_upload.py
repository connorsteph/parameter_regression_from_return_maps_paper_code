import multiprocessing

def worker_function(x):
    num_cpus = multiprocessing.cpu_count()
    print("Number of CPUs in worker process:", num_cpus)

if __name__ == '__main__':
    num_cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=6)
    pool.map(worker_function, range(6))
    pool.close()
    pool.join()
