import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from numba import jit

def get_gaussian_eta(N):
    # Unit length, random angle
    eta = np.random.normal(0,1, N)
    return eta

@jit
def get_particles_array(N):
    positions_x = np.zeros(N)
    positions_y = np.zeros(N)
    angles = np.zeros(N)
    return positions_x, positions_y, angles

@jit
def evolve(M_in, dt, gamma, v0, alpha, nu, N):
    N = int(N)
    M_out = get_particles_array(N)
    rx = M_in[0]
    ry = M_in[1]
    theta = M_in[2]
    eta = get_gaussian_eta(N)
    rx_new = rx + (dt/gamma)*v0*np.cos(theta) + np.sqrt(dt*alpha)*eta
    ry_new = ry + (dt/gamma)*v0*np.sin(theta) + np.sqrt(dt*alpha)*eta
    theta_new = theta + np.sqrt(dt*nu)*eta
    M_out = (rx_new, ry_new, theta_new)
    return M_out

@jit
def calculate_r2(M_in, N):
    positions_x = M_in[0]
    positions_y = M_in[1]
    M_out = (np.power(positions_x, 2), np.power(positions_y, 2))
    xsum = np.sum(M_out[0])
    ysum = np.sum(M_out[1])
    sums = (xsum + ysum)/N
    return sums

@jit
def get_MSD(N, gamma, v0, alpha, nu, dt):
    MSD = np.zeros(N)
    M_fresh = get_particles_array(N)
    for it in range(N):
        new_M = evolve(M_fresh, dt, gamma, v0, alpha, nu, N)
        out_MSD = calculate_r2(new_M, N)
        MSD[it] = out_MSD
    return MSD

def get_several(v0_vec, N, gamma, alpha, nu, dt):
    MSD_tab = []
    for v0 in v0_vec:
        MSD = get_MSD(N, gamma, v0, alpha, nu, dt)
        MSD_tab.append(MSD)
    return MSD_tab

@jit
def get_MSD_2(N, gamma, alpha, nu, dt, v_0):
    positions_x = np.zeros(N)
    positions_y = np.zeros(N)

    velocities = np.zeros(N)

    angles = np.zeros(N)

    msd = np.zeros(N)
    t = 0

    for i in range(N):

        positions_x = positions_x + (dt/gamma)*v_0*np.cos(angles) + np.sqrt(dt * alpha) * get_gaussian_eta(N)
        positions_y = positions_y + (dt/gamma)*v_0*np.sin(angles) + np.sqrt(dt * alpha) * get_gaussian_eta(N)
        angles = angles + np.sqrt(dt * nu) * get_gaussian_eta(N)

        t = t + dt
        temp_x = positions_x**2
        temp_y = positions_y**2
        sums = (np.sum(temp_x) + np.sum(temp_y))/N
        msd[i] = sums
    return msd


def get_MSD_final(v0_vec, N, gamma, alpha, nu, dt):
    mm = []
    for v_0 in v0_vec:
        msd = get_MSD_2(N, gamma, alpha, nu, dt, v_0)
        mm.append(msd)
    return mm
