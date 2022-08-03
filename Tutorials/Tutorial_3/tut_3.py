import pycosat as sat
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
from numpy.random import choice
import multiprocessing as mp
from itertools import repeat
import time

def rand_sat(N, M, k):
    return [np.sort((choice(N, k, replace=False)+1) * choice([-1, 1])).tolist() for m in range(M)]

def solve_cnf(cnf):
    try:
        solved = 0
        for sol in sat.itersolve(cnf):
            print(sol)
            solved += 1
        return solved
    except:
        print("unSAT")
        return None

def solve(cnf):
    s = sat.solve(cnf)
    if s == "UNSAT":
        return 0
    else:
        return 1

def count_parallel(n, F, k, nsamp):
    res = []
    print(f"n = {n}")
    for f in F:
        M = int(f * n)
        cnf_2 = [rand_sat(n, M, k) for m in range(nsamp)]
        val = np.sum([solve(exp) for exp in cnf_2])
        val = val/nsamp
        res.append(val)
    return res


def parallelize(N, K, nsamp):
    for k in K:
        if k == 2:
            F = np.linspace(0.5, 3.0, 30)
        else:
            F = np.linspace(4.0, 6, 30)
        res_tab = []
        p = mp.Pool(processes=8)
        res = p.starmap_async(count_parallel, zip(N, repeat(F), repeat(k), repeat(nsamp))).get()
        res_tab.append(res)
        p.close()
        p.join()
        for res_n, n in zip(res_tab, N):
            print(res_n)


if __name__ == "__main__":
    N = [50, 100, 150]
    K = [2, 3]
    t_Start = time.time()
    nsamp = 100
    parallelize(N, K, nsamp)
    t_end = time.time()
    print("Time elapsed: ", t_end - t_Start)

#%%
