import numpy as np
import pandas as pd
from random import random
import math
import sys
import os
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
from collections import deque


def set_seed(i):
    np.random.seed(i)

# Seed Spreader as described in DBSCAN Revisited
# Junhao Gan and Yufei Tao. "DBSCAN revisited: Mis-claim, un-fixability, and approximation."
# Proceedings of the 2015 ACM SIGMOD international conference on management of data. 2015.
# Junhao Gan and Yufei Tao. "On the Hardness and Approximation of Euclidean DBSCAN",
# ACM Transactions on Database Systems (TODS), vol. 42, no. 3, pp. 1–45, 2017


# obtain n uniformly sampled points within a d-sphere with a fixed radius around a given point. Assigns all points to given cluster
# code partially based on code provided here http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
def random_ball_num(center, radius, d, n, clunum):
    d = int(d)
    n = int(n)
    u = np.random.normal(0, 1, (n, d + 1))  # an array of d normally distributed random variables
    norm = np.sqrt(np.sum(u ** 2, 1))
    r = np.random.random(n) ** (1.0 / d)
    normed = np.divide(u, norm[:, None])
    x = r[:, None] * normed
    x[:, :-1] = center + x[:, :-1] * radius
    x[:, -1] = clunum
    return x

def seedSpreader(n = 2000000, dim=5, ratio_noise=0.001, domain_size=100000, reset_counter= 100, restart_chance_mult = 10,
             radius=100, seed=0, verbose =False, noise_adapt=False, var_density = False):
    """
    Seed Spreader generator function
    :param n: number of data points
    :param dim: dimensionality
    :param ratio_noise: noise ratio
    :param domain_size: domain size
    :param reset_counter: counter for hypersphere points
    :param restart_chance_mult: cluster restart overall
    :param radius: radius of hypersphere
    :param var_density: density based mode introduced in “On the hardness and approximation of euclidean dbscan,”, uses ((i mod 10) + 1) as factor
    :param seed: seed
    :param verbose: verbose
    :param noise_adapt: adapt noise to altered domain size after random walk
    :return: data points as np.array of shape (dim+1, data_num), last column is cluster id
    """

    step = radius/2*dim

    if verbose:
        print("dim: ", dim)
        print("n: ", n)
        print("reset_counter", reset_counter)
        print("base radius: ", radius)
        print("base step:", step)
    set_seed(seed)
    ratio_noise = ratio_noise
    reset_counter = reset_counter
    num_data = round((n* (1-ratio_noise)))
    restart_chance = restart_chance_mult/num_data
    verbose = verbose



    points = []
    pos = np.random.random(dim) * domain_size
    counter = reset_counter
    cluid = 0
    while (len(points) < num_data):

        if var_density:
            factor = (cluid % 10) + 1
        else:
            factor = 1

        rand = np.random.rand()
        if rand < restart_chance:
            if(verbose):
                print("restart occured")
            pos = np.random.random(dim) * domain_size
            counter = reset_counter
            if len(points) > 0:
                cluid += 1
        point = random_ball_num(pos, radius* factor, dim, 1, cluid)
        points.append(point[0])
        counter -= 1

        if counter == 0:
            step_dir = (np.random.random(dim) - 0.5)
            step_dir = step_dir / np.linalg.norm(step_dir)
            #print(pos)
            pos = pos + (step_dir * step * factor)
            counter = reset_counter
            #print(pos)
            if verbose:
                print("step occured")

    points_np = np.array(points)
    mins = []
    maxs = []

    maxall = -1 * np.inf
    minall = np.inf

    for d in range(dim):
        maxs.append(np.max(points_np[:,d]))
        mins.append(np.min(points_np[:, d]))
        if maxall < maxs[d]:
            maxall = maxs[d]
        if minall > mins[d]:
            minall = mins[d]

    noise = np.random.random([n-num_data, dim + 1])

    if noise_adapt=="square":
        dspan = maxall - minall
        for d in range(dim):
            noise[:, d] = (noise[:, d] * dspan * 1.2) + minall - 0.1 * dspan
    elif noise_adapt==True or noise_adapt=="dim":
        for d in range(dim):
            dspan = maxs[d] - mins[d]
            noise[:, d] = (noise[:, d] * dspan * 1.2) + mins[d] - 0.1 * dspan
    else:
        noise = noise*domain_size
    noise[:, -1] = -1

    points.extend(noise)
    return np.array(points)


if __name__ == '__main__':
    dim = 2
    noise_ratio = 0.001
    data = seedSpreader(dim=dim, ratio_noise=noise_ratio, noise_adapt=True, var_density = True)

    print(data)
    datax = data[:, 0:-1]
    print(datax.shape)
    datay = data[:, -1]
    print(datay.shape)

    plt.figure(figsize=(15, 15))
    color = plt.cm.tab20(np.linspace(0, 1, len(np.unique(datay))))
    color = np.append(color, [[0, 0, 0, 1]], axis=0)
    #print(color)
    plt.scatter(datax[:, 0], datax[:, 1], c=color[datay.astype('int32')])
    plt.axis('scaled')
    plt.show()


