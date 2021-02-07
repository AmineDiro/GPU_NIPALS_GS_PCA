# Test multi

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda import cumath
import numpy as np
from time import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from nipals.NIPALS import Nipals_GPU, Nipals_CPU

N_ITER = 5
MIN_SIZE = 500
MAX_SIZE = 1000
n_components = 2

speed = {}

nips_gpu = Nipals_GPU(tol=0.001, maxiter=100, ncomp=n_components)
nips_cpu = Nipals_CPU(tol=0.001, maxiter=100, ncomp=n_components)

for size in [500, 600, 100]:
    time_cpu = np.empty(N_ITER)
    time_gpu = np.empty(N_ITER)
    for it in range(N_ITER):
        X = np.random.randn(size, size)
        #  Fit on GPU :
        t1 = time()
        nips_gpu.fit_on_GPU(X)
        t2 = time()
        time_gpu[it] = t2-t1
        #  Fit on CPU :
        t1 = time()
        nips_cpu.fit(X)
        t2 = time()
        time_cpu[it] = t2-t1
    speed[size] = {
        "mean_time_gpu": np.mean(time_gpu),
        "std_time_gpu": np.std(time_gpu),
        "mean_time_cpu": np.mean(time_cpu),
        "std_time_cpu": np.std(time_cpu),
    }

res = pd.DataFrame(speed)
res = res.transpose()
res.to_csv('result2.csv')
