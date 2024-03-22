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

from pyparsing import nestedExpr

# set seed for data generator
def set_seed(i):
    np.random.seed(i)


# obtain n uniformly sampled points within a d-sphere with a fixed radius around a given point. Assigns all points to
# given cluster code partially based on code provided here:
# http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
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



class densityDataGen:

    def __init__(self, dim=2, clunum=2, clu_ratios=None, core_num=10, min_ratio=0, ratio_noise=0, domain_size=1,
                 radius=1, step=1, ratio_con=0, connections=0, seed=0, dens_factors=False, momentum=0.5,
                 con_momentum=0.9, min_dist=1.1, con_min_dist=0.9, step_spread=0, max_retry=5, verbose=False,
                 safety=True, con_dens_factors=False, con_radius=2, con_step=2, branch=0.05, star=0, square=False,
                 random_start=False):
        """
        For extensive parameter explanations, please consult the GitHub README
        :param dim: dimensionality
        :param clunum: cluster number
        :param clu_ratios: cluster ratios
        :param core_num: overall core number
        :param min_ratio: minimal cluster ratio
        :param ratio_noise: noise ratio
        :param domain_size: domain size
        :param radius: core base radius
        :param step: base random walk step size
        :param ratio_con: connection ratio
        :param connections: connections
        :param seed: seed
        :param dens_factors: cluster density factors
        :param momentum: momentum
        :param con_momentum: connection momentum
        :param min_dist: minimal distance ratio between clusters
        :param con_min_dist: minimal distance ratio between connections
        :param step_spread: spread of randomization of random walk step size
        :param max_retry: maximal number of retrys
        :param verbose: verbose mode
        :param safety: safety mode
        :param con_dens_factors: connection density factors
        :param con_radius: connection radius
        :param con_step: connection random walk step size
        :param branch: branching factor
        :param star: star chance
        :param square: square noise distribution
        :param random_start: random start mode
        """
        set_seed(seed)
        self.verbose = verbose
        self.dim = dim
        self.clunum = clunum
        self.clu_ratios = clu_ratios
        self.core_num = core_num
        self.min_ratio = min_ratio
        self.ratio_noise = ratio_noise
        self.domain_size = domain_size
        self.r_sphere = radius
        self.r_step = step
        self.ratio_con = ratio_con
        self.connections = connections
        self.c_sphere = con_radius
        self.c_step = con_step

        self.con_dens_factors = con_dens_factors
        self.dens_factors = dens_factors
        self.step_spread = step_spread
        self.max_retry = max_retry
        self.momentum = momentum
        self.con_momentum = con_momentum
        self.min_dist = min_dist
        self.con_min_dist = con_min_dist
        self.safety = safety

        self.cores = {}
        self.value_space = []
        self.mins = [domain_size + 1] * self.dim
        self.maxs = [-1] * self.dim

        self.stream_content = []
        self.stream_repeat = []
        self.stream_data_cores_block = {}
        self.stream_num_block = {}
        self.noise_block = {}
        self.cur_stream_point = 0
        self.cur_stream_pos = 0
        self.cur_stream_db = {}

        self.cur_block_num = 0
        self.in_reapeat = False

        self.branch = branch
        self.star = star

        self.square = square
        self.random_start = random_start

        self.data_ratio = 1 - ratio_con - ratio_noise
        if (self.data_ratio <= 0):
            raise BaseException("No data points can be generated")

        if self.clu_ratios == None:
            while (True):
                self.clu_ratios = np.sort(np.random.random(self.clunum - 1) * self.data_ratio)
                self.clu_ratios = np.append(self.clu_ratios, 1)
                dist = 0
                newrun = False
                for i in range(self.clunum):
                    if i==0 and self.clunum == 1:
                        dist = 1
                        self.clu_ratios[i] = self.data_ratio
                    elif i == 0:
                        dist = self.clu_ratios[i]
                    elif i == self.clunum - 1:
                        dist = self.data_ratio - self.clu_ratios[i - 1]
                        self.clu_ratios[i] = self.data_ratio
                    else:
                        dist = self.clu_ratios[i] - self.clu_ratios[i - 1]
                    if dist < self.min_ratio:
                        newrun = True
                        break
                if not newrun:
                    break
        elif (self.clunum == len(self.clu_ratios)):
            prior_ratio = 0
            new_ratio = []
            for ratio in self.clu_ratios:
                ratio = ratio + prior_ratio
                new_ratio.append(ratio)
                prior_ratio = ratio

            self.clu_ratios = np.divide(new_ratio, ratio / self.data_ratio)
        else:
            raise BaseException("Shape of cluster ratio does not match cluster number")

        if self.dens_factors == False:
            self.dens_factors = []
            for _ in self.clu_ratios:
                self.dens_factors.append(1)
        elif self.dens_factors == True:
            self.dens_factors = []
            for _ in self.clu_ratios:
                factor = (np.random.rand() * 1.5) + 0.5
                factor = round(factor, 2)
                # print(factor)
                self.dens_factors.append(factor)
        elif (self.clunum != len(self.dens_factors)):
            raise BaseException("Shape of cluster scale factors does not match cluster number")

        if type(self.branch) is list:
            if (self.clunum != len(self.branch)):
                raise BaseException("Shape of branch chances does not match cluster number")
        elif self.branch == "Rand":
            self.branch = []
            for _ in self.clu_ratios:
                branch = np.random.rand() * 0.5
                self.branch.append(branch)
        else:
            branch = self.branch
            self.branch = [branch] * self.clunum

        if type(self.star) is list:
            if (self.clunum != len(self.star)):
                raise BaseException("Shape of star chances does not match cluster number")
        elif self.star == "Rand":
            # print("I was here")
            self.star = []
            for _ in self.clu_ratios:
                star = np.random.rand() * 0.5
                self.star.append(star)
        else:
            star = self.star
            self.star = [star] * self.clunum

        if (self.verbose):
            print("Cluster size ratios:")
            print(self.clu_ratios)
            print("Cluster scale factors:")
            print(self.dens_factors)

        if type(self.momentum) is list:
            if (self.clunum != len(self.momentum)):
                raise BaseException("Shape of cluster momentum does not match cluster number")
            else:
                if (self.verbose):
                    print("Cluster momentum factors:")
                    print(self.momentum)
        elif self.momentum is None:
            self.momentum = []
            for _ in self.clu_ratios:
                momentum = np.random.rand()
                self.momentum.append(momentum)
            if (self.verbose):
                print("Cluster momentum factors:")
                print(self.momentum)
        else:
            momentum = self.momentum
            self.momentum = [momentum] * self.clunum

        if type(self.core_num) is list:
            if (self.clunum != len(self.core_num)):
                raise BaseException("Shape of cluster core number does not match cluster number")
            else:
                if (self.verbose):
                    print("Cluster core numbers:")
                    print(self.core_num)

        connection_starts = []
        connection_stops = []

        if type(self.connections) is list:
            for con in self.connections:
                # print(con)
                pair = con.split(";")
                # print(pair)
                # print(pair[0])
                connection_starts.append(int(pair[0]))
                connection_stops.append(int(pair[1]))
        else:
            for i in range(self.connections):
                startclu = np.random.choice(len(range(self.clunum)), 1)
                stopclu = np.random.choice(len(range(self.clunum)), 1)
                while (stopclu == startclu):
                    stopclu = np.random.choice(len(range(self.clunum)), 1)
                connection_starts.extend(startclu)
                connection_stops.extend(stopclu)

        if self.con_dens_factors == False:
            self.con_dens_factors = []
            for _ in range(len(connection_starts)):
                self.con_dens_factors.append(1)
        elif self.con_dens_factors == None:
            self.con_dens_factors = []
            for _ in range(len(connection_starts)):
                factor = (np.random.rand() * 1.5) + 0.5
                factor = round(factor, 2)
                self.con_dens_factors.append(factor)
        elif (len(connection_starts) != len(self.con_dens_factors)):
            raise BaseException("Shape of connection scale factors does not match connection number")

        if type(self.con_momentum) is list:
            if (len(connection_starts) != len(self.con_momentum)):
                raise BaseException("Shape of connection momentum does not match connection number")
        elif self.con_momentum == None:
            self.con_momentum = []
            for _ in range(len(connection_starts)):
                con_momentum = np.random.rand()
                self.con_momentum.append(con_momentum)
        else:
            con_momentum = self.con_momentum
            self.con_momentum = [con_momentum] * len(connection_starts)

        if verbose:
            for i in range(len(connection_starts)):
                print("Connection from " + str(connection_starts[i]) + " to " + str(
                    connection_stops[i]) + " with factor " + str(self.con_dens_factors[i]) + " and momentum " + str(
                    self.con_momentum[i]))

        for cluid in range(self.clunum):
            self.generate_cluster(cluid)

        self.connections = {}
        for i in range(len(connection_starts)):
            retry = 0
            conid = -i - 2
            # print("connection " + str(conid))
            while (True):
                noretry = self.make_connection(connection_starts[i], connection_stops[i], conid)
                if noretry:
                    self.connections[conid] = str(connection_starts[i]) + "-" + str(connection_stops[i])

                    if len(self.cores[conid]) == 0:
                        if retry >= self.max_retry:
                            self.cores.pop(conid, None)
                            if (self.verbose):
                                print("Could not generate connection from " + str(connection_starts[i]) + " to " + str(
                                    connection_stops[i]) + " as they were too close")
                            break
                        else:
                            retry += 1
                    else:
                        break
                if retry >= self.max_retry:
                    self.cores.pop(conid, None)
                    if (self.verbose):
                        print("Could not generate connection from " + str(connection_starts[i]) + " to " + str(
                            connection_stops[i]))
                    break
                else:
                    retry += 1

    def generate_cluster(self, cluid):
        """
        generate cluster
        :param cluid: cluster id
        """
        retry = 0
        no_space = self.random_start

        #print(self.cores)
        while (True):

            if cluid == 0 or cluid == 1 or no_space:
                pos = np.random.random(self.dim) * self.domain_size
                if self.verbose:
                    print(f"{cluid}: random start")
            else:
                midids = np.random.choice(range(cluid), 2, replace=False)
                #print(midids)
                #core1 = np.sum(self.cores[midids[0]], axis=0)/len(self.cores[midids[0]])#[np.random.choice(len(self.cores[midids[0]]))]
                #core2 = np.sum(self.cores[midids[1]], axis=0)/len(self.cores[midids[1]])#[np.random.choice(len(self.cores[midids[1]]))]
                core1 = self.cores[midids[0]][np.random.choice(len(self.cores[midids[0]]))]
                core2 = self.cores[midids[1]][np.random.choice(len(self.cores[midids[1]]))]

                pos = (core1 + core2) / 2
                #print(pos)
                if self.verbose:
                    print(f"{cluid}: average start between {midids[0]} and {midids[1]}")
            self.cores[cluid] = []
            noretry = self.generate_cluster_pos(pos, cluid)
            if noretry:
                break
            elif (retry >= self.max_retry and no_space):  # expand domain size, at some point there is enough space for new cluster
                self.domain_size = self.domain_size + self.r_sphere
                retry = 0
                if (self.verbose):
                    print(f"{cluid}: domain size increased")
            elif retry >= self.max_retry and not no_space:
                retry = 0
                no_space = True
                if self.verbose:
                    print(f"{cluid}: restart random")
            else:
                if (self.verbose):
                    print(f"{cluid}: restart")
                retry += 1
        self.cores[cluid] = np.array(self.cores[cluid])

    def make_connection(self, startid, stopid, conid):
        """
        make connection between two given clusters with given conid
        :param startid: id of start cluster
        :param stopid: id of target cluster
        :param conid: id of connection (negative)
        :return: whether connection succeeded
        """
        startcore = self.cores[startid][np.random.choice(len(self.cores[startid]))]
        stopcore = self.cores[stopid][np.random.choice(len(self.cores[stopid]))]

        conind = conid * -1 - 2
        # print("conind: " + str(conind))

        coni_momentum = self.con_momentum
        if type(self.con_momentum) is list:
            coni_momentum = self.con_momentum[conind]

        dist = np.sum((stopcore - startcore) ** 2) ** (0.5)

        pos = startcore
        self.cores[conid] = []
        self.cores[conid].append(pos)
        pos_old = pos

        min_dist = (self.con_dens_factors[conind] * self.c_sphere + self.dens_factors[
            stopid] * self.r_sphere) * self.con_min_dist
        retry_some = False
        tries_some = -1
        while (dist > min_dist):
            # print(dist)
            retry_last = True
            tries_last = 0
            while (retry_last or retry_some):

                if (retry_some):
                    step_dir_old = None

                    if (len(self.cores[conid])) <= 1:
                        startcore = self.cores[startid][np.random.choice(len(self.cores[startid]))]
                        pos = startcore
                        self.cores[conid] = []
                        self.cores[conid].append(pos)
                        pos_old = pos
                    else:
                        old_id = np.random.choice(len(self.cores[conid]))

                        # print(old_id)
                        # print(len(self.cores[conid]))

                        pos_old = self.cores[conid][old_id]

                        self.cores[conid] = self.cores[conid][:old_id]

                    tries_some += 1
                    retry_some = False

                step_dir = (np.random.random(self.dim) - 0.5)
                step_dir = step_dir / np.linalg.norm(step_dir)

                guided = stopcore - pos_old

                step_dir = step_dir * (1 - coni_momentum) + (guided * coni_momentum)
                step_dir = step_dir / np.linalg.norm(step_dir)

                c_step_clu = self.c_step * self.con_dens_factors[conind]
                max_rand = 1.5
                min_rand = 2 / 3
                step_randomness = max(min_rand, min(max_rand, np.random.normal(1, self.step_spread, 1)))
                # print(step_randomness)
                pos = pos_old + step_dir * c_step_clu * step_randomness

                dist = np.linalg.norm(stopcore - pos)

                for c in self.cores[stopid]:
                    dist = min(dist, np.linalg.norm(c - pos))

                attempts = 0

                if (self.tooclose(pos, conid, exclude=[startid, stopid, conid])):
                    # print("too close")
                    if (tries_some >= self.max_retry):
                        return False
                    elif (tries_last >= self.max_retry):
                        retry_some = True
                        tries_last = 0
                    else:
                        tries_last += 1
                else:
                    self.cores[conid].append(pos)
                    pos_old = pos
                    step_dir_old = step_dir
                    retry_last = False

                    for d in range(self.dim):
                        self.mins[d] = min(pos[d] - self.c_sphere * self.con_dens_factors[conind], self.mins[d])
                        self.maxs[d] = max(pos[d] + self.c_sphere * self.con_dens_factors[conind], self.maxs[d])

        connectors = []
        for c in self.cores[conid]:
            if not (self.tooclose_pair(c, conid, startid) or self.tooclose_pair(c, conid, stopid)):
                connectors.append(c)

        if len(connectors) > 0:
            constart = connectors[0]
            target = startcore

            dist = np.linalg.norm(startcore - constart)

            for c in self.cores[startid]:
                cdist = np.linalg.norm(c - constart)
                if dist > cdist:
                    dist = cdist
                    target = c

            guided = constart - target
            step_dir = guided / np.linalg.norm(guided)

            pos = target + (self.con_dens_factors[conind] * self.c_sphere + self.dens_factors[
                startid] * self.r_sphere) * step_dir * self.con_min_dist

            connectors = deque(connectors)
            connectors.appendleft(pos)
            connectors = list(connectors)

            constop = connectors[-1]
            target = stopcore

            dist = np.linalg.norm(stopcore - constop)

            for c in self.cores[stopid]:
                cdist = np.linalg.norm(c - constop)
                if dist > cdist:
                    dist = cdist
                    target = c

            guided = constop - target
            step_dir = guided / np.linalg.norm(guided)

            pos = target + (self.con_dens_factors[conind] * self.c_sphere + self.dens_factors[
                stopid] * self.r_sphere) * step_dir * self.con_min_dist

            connectors.append(pos)

        self.cores[conid] = np.array(connectors)
        return True

    # check if a position is too close to exitsing cores
    def tooclose(self, pos, label, exclude=None, noise=False):
        """
        perform a check regarding closeness
        :param pos: position
        :param label: own label
        :param exclude: labels to exclude
        :param noise: whether pos is intended for noise or a core
        :return: whether too close
        """
        # if (len(points) > 0):
        #    points = points[0]

        #print(self.cores)

        if exclude is None:
            exclude = [label]

        for cluid in self.cores.keys():
            if cluid not in exclude:
                # print(cluid)
                if cluid < -1:
                    factor = self.con_dens_factors[cluid * -1 - 2] * self.c_sphere
                else:
                    factor = self.dens_factors[cluid] * self.r_sphere

                if label < -1:
                    factor += self.con_dens_factors[label * -1 - 2] * self.c_sphere
                elif not noise:
                    factor += self.dens_factors[label] * self.r_sphere
                elif self.safety:
                    factor += factor
                else:
                    factor += 0.1 * factor

                if label < -1:
                    min_distcluid = self.con_min_dist * factor
                else:
                    min_distcluid = self.min_dist * factor
                # print(min_distcluid)
                for core in self.cores[cluid]:
                    dist = 0
                    for i in range(self.dim):
                        dist += (core[i] - pos[i]) ** 2
                        if dist ** 0.5 > min_distcluid:
                            break
                    if dist ** 0.5 < min_distcluid:
                        return True
        return False

    def tooclose_pair(self, pos, label, partner):
        """
        perform pairwise check regarding closeness
        :param pos: position
        :param label: own label
        :param partner: label of partner
        :return: whether too close
        """
        factor1 = 0
        if label >= 0:
            factor1 = self.dens_factors[label] * self.r_sphere
        else:
            factor1 = self.con_dens_factors[label * -1 - 2] * self.c_sphere
        factor2 = 0
        if partner >= 0:
            factor2 = self.dens_factors[partner] * self.r_sphere
        else:
            factor2 = self.con_dens_factors[partner * -1 - 2] * self.c_sphere

        factor = factor1 + factor2

        if label < -1:
            min_distcluid = self.con_min_dist * factor
        else:
            min_distcluid = self.min_dist * factor
        if label < 0 or partner < 0:
            min_distcluid = self.con_min_dist * factor

            # print(str(factor1) + " " + str(factor2) + " " + str(factor) + " " + str(min_distcluid))
        for core in self.cores[partner]:
            dist = 0
            for i in range(self.dim):
                dist += (core[i] - pos[i]) ** 2
                if dist ** 0.5 > min_distcluid:
                    break
            if dist ** 0.5 < min_distcluid:
                return True
        return False

    def generate_cluster_pos(self, start_pos, cluid):
        """
        generate a cluster with given id at a specific location
        :param start_pos: cluster start location
        :param cluid: cluster id
        :return: whether cluster generation succeeded
        """
        if self.tooclose(start_pos, cluid):
            return False

        clu_momentum = self.momentum
        if type(self.momentum) is list:
            clu_momentum = self.momentum[cluid]

        core_num = 0
        if type(self.core_num) is list:
            core_num = self.core_num[cluid]
        else:
            clu_ratio = 0
            if cluid >= 1:
                clu_ratio = self.clu_ratios[cluid] - self.clu_ratios[cluid - 1]
            else:
                clu_ratio = self.clu_ratios[cluid]
            core_num = max(round((clu_ratio) * self.core_num / self.data_ratio), 1)

        if (self.verbose):
            print("Cluster ID " + str(cluid) + " Core Number: " + str(core_num))
        pos = start_pos
        self.cores[cluid].append(pos)
        pos_old = pos
        step_dir_old = None

        branch_chance = self.branch[cluid]
        star_chance = self.star[cluid]

        for i in range(core_num - 1):

            retry_last = True
            tries_last = 0
            retry_some = False
            tries_some = -1
            while (retry_last or retry_some):
                rand = np.random.rand()
                if (retry_some or rand < branch_chance):
                    step_dir_old = None
                    probs = [1 / len(self.cores[cluid])] * len(self.cores[cluid])
                    if star_chance > 0 and len(self.cores[cluid]) > 1:
                        probs = [star_chance]
                        probs.extend([(1 - star_chance) / (len(self.cores[cluid]) - 1)] * (len(self.cores[cluid]) - 1))

                    selected = np.random.choice(len(self.cores[cluid]), p=probs)
                    # print(selected)
                    pos_old = self.cores[cluid][selected]
                    tries_some += 1
                    retry_some = False

                step_dir = (np.random.random(self.dim) - 0.5)
                step_dir = step_dir / np.linalg.norm(step_dir)
                if step_dir_old is not None:
                    step_dir = step_dir * (1 - clu_momentum) + (step_dir_old * clu_momentum)
                step_dir = step_dir / np.linalg.norm(step_dir)
                # print(step_dir)

                r_step_clu = self.r_step * self.dens_factors[cluid]
                max_rand = 1.5
                min_rand = 2 / 3
                step_randomness = max(min_rand, min(max_rand, np.random.normal(1, self.step_spread, 1)))
                # print(step_randomness)
                pos = pos_old + step_dir * r_step_clu * step_randomness
                attempts = 0

                if (self.tooclose(pos, cluid)):
                    if (tries_some >= self.max_retry):
                        return False
                    elif (tries_last >= self.max_retry):
                        retry_some = True
                        tries_last = 0
                    else:
                        tries_last += 1
                else:
                    self.cores[cluid].append(pos)
                    pos_old = pos
                    step_dir_old = step_dir
                    retry_last = False

                    for d in range(self.dim):
                        self.mins[d] = min(pos[d] - self.r_sphere * self.dens_factors[cluid], self.mins[d])
                        self.maxs[d] = max(pos[d] + self.r_sphere * self.dens_factors[cluid], self.maxs[d])

            # while(tooclose(pos, clunum-1, center_store_other, center_clunum, clufactors, (r_sphere*2.5), d)):
            #    if (attempts > 100):
            #        pos_old = startpos
            #        attempts = 0
            #        print("reset to startpos")
            # print("tooclose")
            #    step_dir = np.random.random(d) - 0.5
            #    pos = pos_old + (step_dir/(np.sum(step_dir**2) **(0.5))*np.random.normal(r_step, stepwidth, 1))
            #    attempts = attempts + 1

        return True


    def generate_data(self, data_num, center=False, non_zero=False, seed=None, equal=False):
        """
        Generate data points based on underlying skeleton
        :param data_num: number of data points
        :param center: whether to place a data point at the core center
        :param non_zero: whether cores should be guaranteed to have at least one data point
        :param seed: new seed for data generation
        :param: equal: spread out data points as equal as possible across cluster
        :return: data points as  np.array of shape (dim+1, data_num), last column is cluster id
        """
        testsum = 0
        mins = []
        maxs = []
        data = []

        if seed is not None:
            set_seed(seed)

        con_core_num = 0
        for cluid in self.cores.keys():
            if cluid < -1:
                con_core_num += len(self.cores[cluid])

        for cluid in self.cores.keys():
            cluratio = 0
            # print(self.clu_ratios[cluid])
            cluradius = self.r_sphere * self.dens_factors[cluid]

            if cluid >= 1:
                clu_ratio = self.clu_ratios[cluid] - self.clu_ratios[cluid - 1]
            elif cluid == 0:
                clu_ratio = self.clu_ratios[cluid]
            else:
                clu_ratio = self.ratio_con * len(self.cores[cluid]) / con_core_num
                cluradius = self.c_sphere * self.con_dens_factors[-1 * cluid - 2]

            clu_core_num = len(self.cores[cluid])

            # print(cluid)
            clu_data_num = round(clu_ratio * data_num)
            testsum += clu_data_num
            if equal:
                spread_val = clu_core_num//clu_data_num
                assignment = np.random.choice(clu_core_num, clu_data_num%clu_core_num)
                assignment_counter = Counter(assignment)
            else:
                spread_val = 0
                assignment = np.random.choice(clu_core_num, clu_data_num)
                assignment_counter = Counter(assignment)
            for coreid in range(len(self.cores[cluid])):
                core = self.cores[cluid][coreid]
                core_data_num = assignment_counter[coreid] + spread_val
                if core_data_num == 0 and non_zero:
                    core_data_num = 1
                if center and core_data_num > 0:
                    core_data_num -= 1
                    data.append(core.copy().tolist() + [cluid])
                # print(str(coreid) + " " + str(core_data_num))
                if core_data_num > 0:
                    data_new = random_ball_num(core, cluradius, self.dim, core_data_num, cluid)
                    data.extend(data_new.tolist())

        noisenum = max(round(data_num * self.ratio_noise), data_num - len(data))
        noise = np.random.random([noisenum, self.dim + 1])

        maxall = -1 * np.inf
        minall = np.inf
        if self.square:
            for d in range(self.dim):
                if maxall < self.maxs[d]:
                    maxall = self.maxs[d]
                if minall > self.mins[d]:
                    minall = self.mins[d]
            dspan = maxall - minall
            for d in range(self.dim):
                noise[:, d] = (noise[:, d] * dspan * 1.2) + minall - 0.1 * dspan
        else:
            for d in range(self.dim):
                dspan = self.maxs[d] - self.mins[d]

                noise[:, d] = (noise[:, d] * dspan * 1.2) + self.mins[d] - 0.1 * dspan

        noise[:, -1] = -1
        truenoise = []
        for n in noise:
            while (self.tooclose(n, -1, noise=True)):
                n = np.random.random([self.dim + 1])
                for d in range(self.dim):
                    dspan = self.maxs[d] - self.mins[d]
                    n[d] = (n[d] * dspan * 1.2) + self.mins[d] - 0.1 * dspan
                    n[-1] = -1
            truenoise.append(n)

            # print("noise removed")

        data.extend(truenoise)
        # print(len(data))

        if not non_zero:
            while (len(data) > data_num):
                data.pop()
                print(len(data))

        return np.array(data)

    def paint(self, dim1, dim2, data=None, show_radius=True, show_core=True, cores=None):
        """
        Data Generator Painter
        :param dim1: first dimension of plot
        :param dim2: second dimension of plot
        :param data: data points to draw
        :param show_radius: whether to show the core radii
        :param show_core: whether to show the core positions
        :param cores: if only specific core are supposed to be drawn
        """
        if cores is None:
            cores = self.cores

        num_col = max(self.cores.keys()) - min(0, min(self.cores.keys())) + 2
        color = plt.cm.tab20(np.linspace(0, 1, num_col))
        plt.figure(figsize=(15, 15))
        if data is not None:
            plt.scatter(data[:, dim1], data[:, dim2], color=color[data[:, len(data[0]) - 1].astype('int32') + 1],
                        alpha=1)

        legend = []

        for cluid in cores.keys():
            if len(cores[cluid]) > 0:
                if self.verbose:
                    if cluid >= 0:
                        patch = mpatches.Patch(color=color[cluid + 1], label=cluid)
                        legend.append(patch)
                    else:
                        patch = mpatches.Patch(color=color[cluid + 1], label=self.connections[cluid])
                        legend.append(patch)
                # print(self.cores[cluid])
                if show_core:
                    if data is None:
                        plt.scatter(cores[cluid][:, dim1], cores[cluid][:, dim2], color=color[cluid + 1], alpha=1)
                    else:
                        plt.scatter(cores[cluid][:, dim1], cores[cluid][:, dim2], color='black', alpha=1)
                if show_radius:
                    for core in cores[cluid]:
                        # print(core[0])
                        if cluid >= 0:
                            plt.gca().add_patch(mpatches.Circle((core[dim1], core[dim2]),
                                                                radius=self.r_sphere * self.dens_factors[cluid],
                                                                color=color[cluid + 1], alpha=0.2))
                        else:
                            plt.gca().add_patch(mpatches.Circle((core[dim1], core[dim2]),
                                                                radius=self.c_sphere * self.con_dens_factors[
                                                                    (-1 * cluid) - 2], color=color[cluid + 1],
                                                                alpha=0.2))

        if data is not None:
            if -1 in data[:, len(data[0]) - 1]:
                patch = mpatches.Patch(color=color[0], label="Noise")
                legend.append(patch)

        plt.axis('scaled')

        #dspan1 = self.maxs[dim1] - self.mins[dim1]
        #dspan2 = self.maxs[dim2] - self.mins[dim2]
        #plt.xlim(self.mins[dim1] - 0.1 * dspan1, self.maxs[dim1] + 0.1 * dspan1)
        #plt.ylim(self.mins[dim2] - 0.1 * dspan2, self.maxs[dim2] + 0.1 * dspan2)

        plt.ylabel(dim2 + 1)
        plt.xlabel(dim1 + 1)
        if self.verbose:
            plt.legend(handles=legend)
            # plt.ylim(50,100)
            # plt.xlim(0,50)
        plt.show()

    def display_data(self, data, show_radius=False, show_core=False, dims=None, dcount=2):
        """
        Display data set
        :param data: data set
        :param show_radius: whether to show the core radii
        :param show_core: whether to show the core positions
        :param dims: specific dimensions to show (if there are more than 2)
        :param dcount: amount of dimensions to display pairs of (if no specific dimensions are given)
        """
        if (self.dim == 2):
            self.paint(0, 1, data=data, show_radius=show_radius, show_core=show_core)
        elif dims is not None:
            for d1 in dims:
                for d2 in dims:
                    if d1 < d2:
                        self.paint(d1, d2, data=data, show_radius=show_radius, show_core=show_core)
        else:
            diff = []
            for d in range(self.dim):
                diff.append(self.maxs[d] - self.mins[d])

            # print(diff)
            dims = np.argsort(diff)[:dcount]
            # print(dims)
            for d1 in dims:
                for d2 in dims:
                    if d1 < d2:
                        self.paint(d1, d2, data=data, show_radius=show_radius, show_core=show_core)

    def display_cores(self, dims=None, dcount=2):
        """
        Display data generator skeleton
        :param dims: specific dimensions to show (if there are more than 2)
        :param dcount: amount of dimensions to display pairs of (if no specific dimensions are given)
        """
        # print(self.cores)
        if (self.dim == 2):
            self.paint(0, 1)
        elif dims is not None:
            for d1 in dims:
                for d2 in dims:
                    if d1 < d2:
                        self.paint(d1, d2)
        else:
            diff = []
            for d in range(self.dim):
                diff.append(self.maxs[d] - self.mins[d])

            # print(diff)
            dims = np.argsort(diff)[:dcount]
            # print(dims)
            for d1 in dims:
                for d2 in dims:
                    if d1 < d2:
                        self.paint(d1, d2)

    def display_cores_selected(self, cores, dims=None, dcount=2):
        """
        Display parts of the data generator skeleton
        :param cores: which cores to display
        :param dims: specific dimensions to show (if there are more than 2)
        :param dcount: amount of dimensions to display pairs of (if no specific dimensions are given)
        """
        if (self.dim == 2):
            self.paint(0, 1, cores=cores)
        elif dims is not None:
            for d1 in dims:
                for d2 in dims:
                    if d1 < d2:
                        self.paint(d1, d2, cores=cores)
        else:
            diff = []
            for d in range(self.dim):
                diff.append(self.maxs[d] - self.mins[d])

            # print(diff)
            dims = np.argsort(diff)[:dcount]
            # print(dims)
            for d1 in dims:
                for d2 in dims:
                    if d1 < d2:
                        self.paint(d1, d2, cores=cores)

    def toggle_verbose(self):
        """
        swaps verbose setting
        """
        self.verbose = not self.verbose

    def init_stream(self, command="", default_duration=1000):
        """
        Initialize stream based on strong command
        :param command: String command for stream
        :param default_duration: default duration of a stream block
        :return: stream settings:  stream content, which part repeats, data_cores_block, num_block, noise_block
        """
        # command_repeats = nestedExpr('[',']').parseString('[' + str(command) + ']').asList()

        command = command + " "
        block = 0
        loopcount = 0
        loopstart = {}
        loopstop = {}

        curLoopId = 0

        data_cores_block = {}
        data_cores_block[0] = {}
        num = {}

        noise_block = {0: 0}
        num_block = {}

        nested_loop = {}

        loopstart[0] = 0

        i = 0
        while i < len(command):
            # print(i)
            # print(command[i])

            if command[i] == "(":
                curLoopId += 1
                loopstart[curLoopId] = block
                # print("start seen " + str(curLoopId))

            elif command[i] == ")":

                curLoopStopId = len(loopstart) - 1
                while curLoopStopId in loopstop.keys():
                    curLoopStopId -= 1

                loopstop[curLoopStopId] = block
                if i < len(command) - 1:
                    if command[i + 1] == "{":
                        j = 2
                        num_loop = ""
                        while command[i + j] != "}":
                            num_loop = num_loop + command[i + j]
                            j += 1
                        num_loop = int(num_loop)
                        i = i + j
                        num[curLoopStopId] = num_loop
                    else:
                        num[curLoopStopId] = 'x'
                else:
                    num[curLoopStopId] = 'x'
            elif command[i] == "|":
                if command[i + 1] == "{":
                    j = 2
                    c_block = ""
                    while command[i + j] != "}":
                        c_block = c_block + command[i + j]
                        j += 1
                    c_block = int(c_block)
                    i = i + j
                    num_block[block] = c_block
                else:
                    num_block[block] = default_duration

                block += 1
                data_cores_block[block] = {}
                noise_block[block] = 0

            elif command[i] == "n":
                noise_block[block] = 1

            elif command[i].isnumeric() or command[i] == "-":
                cid = command[i]
                if command[i] == "-":
                    cid += command[i + 1]
                    i += 1
                while command[i + 1].isnumeric():
                    cid = cid + command[i]
                    i += 1
                cid = int(cid)
                if command[i + 1] == "c":
                    i += 2
                    eid = command[i]
                    while command[i + 1].isnumeric():
                        eid = eid + command[i]
                        i += 1
                    eid = int(eid)

                    for conkey in self.connections.keys():
                        if (self.connections[conkey] == str(cid) + "-" + str(eid) or self.connections[conkey] == str(
                                eid) + "-" + str(cid)):
                            cid = conkey

                # print(self.cores.keys())
                # print(command[i+1])

                if (command[i + 1] == "["):
                    j = 2
                    # print("I was here")
                    # print(block)
                    # print(cid)

                    # print(data_cores_block[block])
                    data_cores_block[block][cid] = []
                    # print(data_cores_block[block][cid])

                    while command[i + j] != "]":
                        if command[i + j] != ",":
                            cstartid = command[i + j]
                            while command[i + j + 1].isnumeric():
                                cstartid = cstartid + command[i + j + 1]
                                j += 1
                            cstartid = int(cstartid)

                            if command[i + j + 1] == ":":
                                j += 2
                                cstopid = command[i + j]

                                while command[i + j + 1].isnumeric():
                                    cstopid = cstopid + command[i + j + 1]
                                    j += 1
                                cstopid = int(cstopid)
                                data_cores_block[block][cid].extend(np.arange(cstartid, cstopid + 1).tolist())
                            else:
                                data_cores_block[block][cid].append(cstartid)
                        j += 1
                    i = i + j
                else:
                    data_cores_block[block][cid] = np.arange(len(self.cores[cid])).tolist()
            i += 1

        loopstop[0] = block
        num[0] = 1

        if block not in num_block.keys():
            num_block[block] = default_duration

        for l in range(len(loopstart)):
            startl = loopstart[l]
            stopl = loopstop[l]
            l2 = l + 1
            nested_loop[l] = []
            while l2 in range(len(loopstart)) and loopstart[l2] <= stopl:
                nested_loop[l].append(l2)
                l2 += 1
        for l in nested_loop.keys():
            prune = []
            for l2 in nested_loop[l]:
                for l3 in nested_loop[l]:
                    if l3 in nested_loop[l2]:
                        prune.append(l3)
            update = []
            for l2 in nested_loop[l]:
                if l2 not in prune:
                    update.append(l2)
            nested_loop[l] = update

        content = {}
        remaining_loops = len(nested_loop)

        print(num)
        for i in num.keys():
            if len(num.keys()) < 2:
                break
            elif i != list(num.keys())[-2]:
                if num[i] == 'x':
                    num[i] = 2
            else:
                if num[i] == 'x':
                    if loopstop[i] != block:
                        num[0] = 'x'
                        num[i] = 2

        # print(loopstart)
        # print(loopstop)
        # print(num)
        print(data_cores_block)
        # print(num_block)
        # print(noise_block)
        # print(nested_loop)

        while remaining_loops > 0:

            # print(remaining_loops)
            for lk in nested_loop.keys():
                if len(nested_loop[lk]) > 0 and not lk in content.keys():
                    wait = False
                    for lk2 in nested_loop[lk]:
                        if not lk2 in content.keys():
                            wait = True
                    if not wait:
                        startb = loopstart[lk]
                        stopb = loopstop[lk]
                        content[lk] = []
                        for lk2 in nested_loop[lk]:
                            repeats = 1
                            content[lk].extend(np.arange(startb, loopstart[lk2]).tolist())
                            if num[lk2] != 'x':
                                repeats = num[lk2]
                            for j in range(repeats):
                                content[lk].extend(content[lk2])
                            startb = loopstop[lk2] + 1
                        remaining_loops -= 1
                        if startb < stopb + 1:
                            content[lk].extend(np.arange(startb, stopb + 1).tolist())

                        # print(str(lk) + " done: " + str(remaining_loops))
                else:
                    if not lk in content.keys():
                        content[lk] = np.arange(loopstart[lk], loopstop[lk] + 1).tolist()
                        remaining_loops -= 1
                        # print(str(lk) + " done: " + str(remaining_loops))

        # print(content)

        repeat = ""
        if not len(num.keys()) < 2:
            finalloop = list(num.keys())[-2]
            if num[finalloop] == 'x':
                repeat = content[finalloop]
            elif loopstop[finalloop] != block:
                repeat = [block]
            else:
                repeat = content[0]
        else:
            repeat = [block]

        # print(final)

        self.stream_content = content[0]
        self.stream_repeat = repeat
        self.stream_data_cores_block = data_cores_block
        self.stream_num_block = num_block
        self.noise_block = noise_block

        self.cur_stream_point = 0
        self.cur_stream_pos = 0
        # self.prev_block_num = 0
        self.in_repeat = False
        self.cur_block_num = 0
        self.cur_stream_dist = {}
        if len(num_block) > 0:
            self.cur_block_num = num_block[0]
            self.cur_stream_dist = {}

        return content[0], repeat, data_cores_block, num_block, noise_block

    def display_stream(self, command="", default_duration=1000, show_core=True, show_radius=True):
        """
        Display the behaviour of given stream command String (also sets the command to be the current stream)
        :param command: String command for stream
        :param default_duration: default duration of a stream block
        :param show_radius: whether to show the core radii
        :param show_core: whether to show the core positions
        """
        self.init_stream(command=command, default_duration=default_duration)
        self.display_current_stream(show_core, show_radius)

    def display_current_stream(self, show_core=True, show_radius=True):
        """
        Display the behaviour of the currently set stream command
        :param show_radius: whether to show the core radii
        :param show_core: whether to show the core positions
        """
        points = 0

        for block in self.stream_content:
            # print(block)
            # print(data_cores_block[block])
            # print(num_block[block])

            block_cores = {}
            for cluid in self.stream_data_cores_block[block].keys():
                block_cores[cluid] = self.cores[cluid][self.stream_data_cores_block[block][cluid]]
            self.display_cores_selected(block_cores,show_core,show_radius)

    def __iter__(self):
        """
        :return: return stream iterator object (itself)
        """
        return self

    def __next__(self):
        """
        :return: next data point based on current stream state
        """
        recalc_db = False

        if self.cur_stream_point == 0 and self.cur_stream_pos == 0:
            recalc_db = True

        self.cur_stream_point += 1

        if not self.in_repeat:
            block = self.stream_content[self.cur_stream_pos]

            if self.cur_stream_point > self.cur_block_num:
                self.cur_stream_pos += 1
                self.cur_stream_point = 1

                if self.cur_stream_pos >= len(self.stream_content):
                    self.in_repeat = True
                    self.cur_stream_pos = 0
                    block = self.stream_repeat[self.cur_stream_pos]
                    print("repeat")
                else:
                    print("generates from block: " + str(self.cur_stream_pos))
                    block = self.stream_content[self.cur_stream_pos]
                self.cur_block_num = self.stream_num_block[block]
                recalc_db = True
        else:
            block = self.stream_repeat[self.cur_stream_pos]
            if self.cur_stream_point > self.cur_block_num:
                self.cur_stream_pos += 1
                self.cur_stream_point = 1

                if self.cur_stream_pos >= len(self.stream_repeat):
                    self.cur_stream_pos = 0
                block = self.stream_repeat[self.cur_stream_pos]
                self.cur_block_num = self.stream_num_block[block]
                recalc_db = True
        if recalc_db:
            block_db = {}

            con_core_num = 0
            for cluid in self.stream_data_cores_block[block].keys():
                if cluid < -1:
                    con_core_num += len(self.stream_data_cores_block[block][cluid])

            db_sum = 0
            for cluid in self.stream_data_cores_block[block].keys():
                if cluid >= 1:
                    block_db[cluid] = self.clu_ratios[cluid] - self.clu_ratios[cluid - 1]
                elif cluid == 0:
                    block_db[cluid] = self.clu_ratios[cluid]
                else:
                    block_db[cluid] = self.ratio_con * len(self.stream_data_cores_block[block][cluid]) / con_core_num
                db_sum += block_db[cluid]

            for cluid in block_db.keys():
                block_db[cluid] = (block_db[cluid] / db_sum) * (1 - self.noise_block[block] * self.ratio_noise)

            if self.noise_block[block] == 1:
                block_db[-1] = self.ratio_noise
            self.cur_stream_db = block_db

        cluid_choice = np.random.choice(list(self.cur_stream_db.keys()), 1, p=list(self.cur_stream_db.values()))[0]

        # self.cores[cluid][self.stream_data_cores_block[block][cluid]]
        if cluid_choice == -1:
            n = np.random.random([self.dim + 1])
            for d in range(self.dim):
                dspan = self.maxs[d] - self.mins[d]
                n[d] = (n[d] * dspan * 1.2) + self.mins[d] - 0.1 * dspan
                n[-1] = -1
            while (self.tooclose(n, -1, noise=True)):
                n = np.random.random([self.dim + 1])
                for d in range(self.dim):
                    dspan = self.maxs[d] - self.mins[d]
                    n[d] = (n[d] * dspan * 1.2) + self.mins[d] - 0.1 * dspan
                    n[-1] = -1
            return n
        else:
            core_choice = np.random.choice(self.stream_data_cores_block[block][cluid_choice])
            core_chosen_pos = self.cores[cluid_choice][core_choice]
            cluradius = 0
            if cluid_choice >= 0:
                cluradius = self.r_sphere * self.dens_factors[cluid_choice]
            else:
                cluradius = self.c_sphere * self.con_dens_factors[-1 * cluid_choice - 2]
            return random_ball_num(core_chosen_pos, cluradius, self.dim, 1, cluid_choice)[0]
