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

def seedSpreader(n = 2000000, dim=5, ratio_noise=0.001, domain_size=100000, reset_counter= 100, reset_chance_mult = 10,
             radius=100, shift=250, seed=0, verbose =False, noise_adapt=False):
    if verbose:
        print("dim: ", dim)
        print("n: ", n)
        print("reset_counter", reset_counter)
        print("radius: ", radius)
        print("shift:", shift)
    set_seed(seed)
    ratio_noise = ratio_noise
    reset_counter = reset_counter
    num_data = round((n* (1-ratio_noise)))
    reset_chance = reset_chance_mult/num_data
    verbose = verbose



    points = []
    pos = np.random.random(dim) * domain_size
    counter = reset_counter
    cluid = 0
    while (len(points) < num_data):
        rand = np.random.rand()
        if rand < reset_chance:
            if(verbose):
                print("reset occured")
            pos = np.random.random(dim) * domain_size
            counter = reset_counter
            if len(points) > 0:
                cluid += 1
        point = random_ball_num(pos, radius, dim, 1, cluid)
        points.append(point[0])
        counter -= 1

        if counter == 0:
            shift_dir = (np.random.random(dim) - 0.5)
            shift_dir = shift_dir / np.linalg.norm(shift_dir)
            #print(pos)
            pos = pos + shift_dir * shift
            counter = reset_counter
            #print(pos)
            if verbose:
                print("shift occured")

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
    shift = 50*dim
    data = seedSpreader(dim=2, shift=shift, verbose=True, noise_adapt=True)

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


