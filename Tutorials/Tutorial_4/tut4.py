import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from numba import jit

@jit
def get_gaussian_eta():
    # Unit length, random angle
    eta = np.random.normal(0,1, 3)
    return eta

@jit
def get_particles_array(N):
    pos = np.zeros((int(N), 3))
    return pos

@jit
def evolve(M, dt, gamma, v0, alpha, nu):
    out_M = np.zeros(M.shape)
    for i in range(M.shape[0]):
        rx, ry, theta = M[i]
        eta = get_gaussian_eta()
        rx_new = rx + dt/gamma*v0*np.cos(theta) + np.sqrt(dt*alpha)*eta[0]
        ry_new = ry + dt/gamma*v0*np.sin(theta) + np.sqrt(dt*alpha)*eta[0]
        theta_new = theta + np.sqrt(dt*nu)*eta[2]
        tmp = [rx_new, ry_new ,theta_new]
        out_M[i] = tmp
    return out_M

@jit
def calculate_r2(M, N):
    num = M.shape[0]
    out = np.zeros(num)
    for i in range(num):
        out[i] = np.power((np.sqrt(np.power(M[i][0], 2) + np.power(M[i][1], 2)) - np.power(M[0][0], 2) + np.power(M[0][1], 2)) ,2)
    out_MSD = np.divide(np.sum(out), N)
    return out_MSD

@jit
def get_MSD(N, gamma, v0, alpha, nu, t_tot):
    MSD = np.zeros(t_tot.size)
    M_beg = get_particles_array(N)
    for it, dt in enumerate(t_tot):
        M = get_particles_array(N)
        new_M = evolve(M, dt,gamma, v0, alpha, nu)
        num = M.shape[0]
        out = np.zeros(num)
        for i in range(num):
            out[i] = np.power((np.sqrt(np.power(new_M[i][0], 2) + np.power(new_M[i][1], 2)) - np.power(M_beg[i][0], 2) + np.power(M_beg[i][1], 2)) ,2)
        out_MSD = np.divide(np.sum(out), N)
        MSD[it] = out_MSD
    return MSD

def get_several(v0_vec, N, gamma, alpha, nu, t_tot):
    MSD_tab = []
    for v0 in v0_vec:
        MSD = get_MSD(N, gamma, v0, alpha, nu, t_tot)
        MSD_tab.append(MSD)
    return MSD_tab