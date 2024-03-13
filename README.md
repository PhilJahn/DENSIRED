# DENSIRED

## How to use

Use the following code to generate a skeleton. Parameters are listed below.

```
skeleton = datagen.densityDataGen()
```

Use the following code to obtain a dataset with *n* points from a skeleton. There are additional parameters, that are also listed below.

```
data = skeleton.generate_data(n)
datax = data[:,0:-1]
datay = data[:,-1]
```

To visualize the skeleton, call the following code. For higher-dimensionalities, either specify *dcount* to get all pairs of the *dc* most spread out dimensions or specify the desired dimensions directly with *dims*.
```
skeleton.display_cores(dims=[d1,d2,...], dcount=dc)
```

To visualize a dataset, call the following. Give the dataset as *data*. The flags *show_radius* and *show_core* decide whether to display the core radii and core centers, respectively. For higher-dimensionalities, as with dispaly_cores, either specify *dcount* to get all pairs of the *dc* most spread out dimensions or specify the desired dimensions directly with *dims*.
```
skeleton.display_data(data, show_radius=False, show_core=False, dims=[d1,d2,...], dcount=dc)
```

To initialize a stream, call the following function. The *command*-String controls the stream behavior. The *default_duration* is the default duration of a block of the stream. default_duration does not need to be specified, in which case it has a value of 1000. The *command*-String will be explained in more detail further below.
```
skeleton.init_stream(command=commandstring, default_duration = 1000)
```

In order to get an element from the stream, just use the skeleton as an iterator
```
x = skeleton.next()
```

To visualize a data stream, call the following.
```
skeleton.display_current_stream()
```

Alternatively, use this to set a stream command and display it in one command. Parameters are analogous to *init_stream*.
```
skeleton.display_stream(command=commandstring, default_duration = 1000)
```


## Skeleton Parameters
| **Parameter**        | **Abrev.** | **Default** | **Role**                                                                                                                                                                                                                                                                      |
|----------------------|------------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dim                  | *dim*     | $2$           | dimensionality of the data set                                                                                                                                                                                                                                                |
| clunum               | *k*    | $2$           | number of clusters                                                                                                                                                                                                                                                            |
| core_num            | *cln*   | $10$          | number of cores across all clusters allows for explicit specification of core number per cluster                                                                                                                                                                              |
| domain_size         | *ds*   | $1$         | initial range of the starting position of the clusters                                                                                                                                                                                                                        |
| radius               | *r*   | $1$         | base range around core center where points can be generated                                                                                                                                                                                                                   |
| step                | $\delta$     | $1$         | base distance between cores                                                                                                                                                                                                                                                   |
| dens_factors       |  *cf*   | False       | cluster density/scale factors<be>multiplies both step size and radius for cluster<br> assigns all clusters a density factor $1$ if False<br> assign random dens_factors *cf*$_{i}$ between $0.5$ and $2$ if True<br>can also be set per cluster in an array                                                                                                                  |
| momentum           | $\omega$     | $0.5$         | cluster momentum factors<br>assigns the value to all clusters as stickiness factor assigns<br>sets random stickiness factors between $0$ and $1$ if None<br>can also be set per cluster in an array                                                                                                                |
| min_dist            | *o*   | $1.1$       | overlap factor on the minimal distance between cores (sum of core sizes)                                                                                                                                                                                                              |
| step_spread | *w*     | $0$           | width of normal distribution on shift                                                                                                                                                                                                                                         |
| max\_retry           | *mr*  | $5$           | maximal number of attempts for generation before generation enters the failure handling                                                                                                                                                                                       |
| branch               | $\beta$   | $0.05$           | chance of creating a branch from the prior scaffolding of the cluster<br>done by restarting from a randomly selected core of the current cluster                                                                                                                                 |
| star                 |  $\varkappa$     | $0$           | chance of the initial starting core being chosen for any attempt of restarting applies to both failure and branching<br>remaining cores all have an equal probability                                                                                                                                                                    |
| seed                 |   *i*    | $0$         | seed for random operations                                                                                                                                                                                                                                                    |
| connections          | *cc*       | $0$         | number of connections randomly picks the specified number of connection pairs<br>allows for explicit specification of desired connections if a list is given<br>connections have to specified as "startid;endid"<br>only guarantees number of attempted connections, not final number |
| con_radius          | *r*$_c$    | $2$         | analogous to radius, used for connections                                                                                                                                                                                                                                     |
| con_step           | $\delta_c$  | $2$         | analogous to step, used for connections                                                                                                                                                                                                                                      |
| con_dens_factors  | *cf*$_c$ | False       | analogous to dens_factors, used for connections                                                                                                                                                                                                                             |
| con_momentum     | $\omega_c$  | $0.9$       | analogous to momentum, used for connections                                                                                                                                                                                                                                 |
| con_min_dist       | *o*$_c$  | $0.9$       | factor on the minimal distance between a connection and other cores                                                                                                                                                                                                           |
| clu\_ratios          | *ra*    | None        | distribution of data points across clusters                                                                                                                                                                                                                                   |
| min\_ratio           | *ra*$_m$   | $0$           | minimal ratio of cores/points<br>(only used when these are randomly determined through *ra*, should be $<<1$)                                                                                                                                                                   |
| ratio\_noise         | *ra*$_n$     | $0$         | ratio of noise data points                                                                                                                                                                                                                                                    |
| ratio\_con           | *ra*$_c$    | $0$         | ratio of connection data points                                                                                                                                                                                                                                               |
| square               | $\square$       | False       | generate noise in square data space                                                                                                                                                                                                                                           |
| random_start               | *rs*       | False       | whether to start at a random position (True) or to start between two different clusters’ random cores (False)                                                                                                                                                                                                                                           |
| verbose              | *v*       | False       | Generate text outputs at key events                                                                                                                                                                                                                                           |
| safety               | *s*       | True        | forces noise generation outside of two times core radius      |
## Data Generator Parameters
| **Parameter**        | **Abrev.** | **Default** | **Role**                                                                                                                                                                                                                                                                      |
|----------------------|------------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| data_num          | *n*      |             | number of data points                                                                                                                                                      |
| non_zero          | *d_nz*       |     False        | guarantee at least one point per core                                                                                                                                                         |
| center         | *d_c*       |     False        | place first data point of core at center of core                                                                                                                                                         |
| seed         | *d_i*       |     None        | Updates generator seed<br>if None, maintains current data/skeleton generator seed                                                                                                                                                 |

## DENSIRED for Streams
The core skeleton provided by DENSIRED allows for a setup for the stream data and offers user control over changes on a cluster and even a sub-cluster level. 
A simple continuous random sampling of new data points from all clusters according to the specified data ratios is, of course, possible. 
However, DENSIRED offers the user the ability to specify which clusters and even which cluster core to draw from.

To do so, the user has to specify a string parameter that describes the desired behavior of the data stream.
The generator supports one stream at a time and can be called as an iterator to get the next value from the stream. 
If the specification string is changed, the stream is reset back to the start.

The stream is split into specific blocks. These blocks are separated by "|" in the string. 
The stream iterator remains inside a block for a set amount of data points (either based on a default value or a specified number). 
The number of points to generate is specified by using a numeric value 𝑥 between braces "{x}".
Each block has a user-defined set of clusters and cores that the new data point can be drawn from. 

The clusters are set by their numeric identifier, separated from other clusters by "#". 
Within each cluster, individual cores can be specified. This is done by denoting the core identifiers inside square brackets "[x]". 
This supports a comma-separated list of individual identifiers, an end-inclusive range of identifiers denoted by a colon, as well as a combination of both.

By using regular brackets, it is possible to denote repetitions as well. These can be nested and their repetition number is designated with braces as with the point numbers for individual blocks. 
If no number is specified, the repetition count defaults to repeating the loop content once, meaning that the sequence will occur twice.
The exception to this is when the repetition without a repetition number is the final part of the string. In that case, the loop is instead repeated endlessly. If there is still a block at the end of the string after the last repetition, the stream stays in that block.
Should the final element be a repetition with a fixed count, the entire stream sequence is looped instead to maintain the repetition number.

Points are generated from the cores of the current block. The weights associated are drawn from the given parameters. 
The noise ratio is maintained if is added to a block to designate the section as a noisy one. 

If a connection is requested by calling its identifier, it also follows the connection ratio, with the ratio being split between all connections based on their core numbers during the block. 
The clusters are weighted according to the cluster ratios when disregarding any cluster that does not have a core in the block. There is no reduced weight for clusters with fewer cores.

To visualize the stream and to easily make adjustments, it is possible to request the data generator to display the behavior for the stream, which will provide a visualization of the skeleton of the block in each overall stream step.
It is especially advisable to display the stream behavior before generating the data when individual cores were chosen to ensure the desired cores were selected


## Reproducibility ##


The 'high' datasets (datasets/high_data_{dim}.npy), which were used in the paper, were generated using the following code. The results for the various algorithms on the 'high' datasets used for Table 3 and Figure 3 can be found in results/high_data_results.
```
for dim in [2,5,10,50,100]:
    x = datagen.densityDataGen(dim=dim, ratio_noise = 0.1, max_retry=5, dens_factors=[1,1,0.5, 0.3, 2, 1.2, 0.9, 0.6, 1.4, 1.1], square=True, 
                       clunum= 10, seed = 6, core_num= 200, momentum=0.8, step=1.5,
                       branch=0.1, star=1, verbose=False, safety=False, domain_size = 20, random_start=False)
    data = x.generate_data(5000)
    datax = data[:,0:-1]
    max = np.max(datax) - np.min(datax)
    for d in range(len(datax[0])):
        datax[:,d] = (datax[:,d] - np.min(datax[:,d]))/max
    datay = data[:,-1]
```

The results of the intrinsic dimensionality benchmarking used for Fig.5 are in intrinsic_dim_100_500_1000_merged.csv. The data for it was generated by the following code for all intseed $\in\[0:9\]$
```
dim = 100
for core_num in [100, 500, 1000, 5000, 10000]:
  for step in [1, 1.25, 1.5, 1.75, 2]:
    for branch in [0, 0.0001, 0.001, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
      for star in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        for momentum in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 0.99, 1]:
          x = datagen.densityDataGen(dim=dim, ratio_noise=0, max_retry=5, dens_factors=1, square=True,
                                     clunum=1, seed=intseed, core_num=core_num, momentum=momentum,
                                     branch=branch, star=star, step=step,
                                     verbose=False, safety=False, domain_size=20, random_start=False)
          data = x.generate_data(5000*core_num/100)
          datax = data[:, 0:-1]
          datax = _norm_data(datax)
          pca = PCA().fit(datax)
          cumsum = np.cumsum(pca.explained_variance_ratio_)
          int_dim90 = np.argwhere(cumsum >= 0.9)[0][0] + 1
          int_dim95 = np.argwhere(cumsum >= 0.95)[0][0] + 1
          int_dim99 = np.argwhere(cumsum >= 0.99)[0][0] + 1
          int_dim75 = np.argwhere(cumsum >= 0.75)[0][0] + 1
          int_dim50 = np.argwhere(cumsum >= 0.50)[0][0] + 1
          int_dim25 = np.argwhere(cumsum >= 0.25)[0][0] + 1
```

The unused 2-d evaluation dataset (datasets/new_data_scaled.npy) was generated with the following code. An unscaled version from directly after the generate_data call is included as datasets/new_data.npy.

```
x = datagen.densityDataGen(dim=2, ratio_noise = 0.1, max_retry=5, dens_factors=[1,1,0.5, 0.3, 2, 1.2, 0.9, 0.6, 1.4, 1.1], square=True, 
                   clunum= 10, seed = 6, core_num= 200, momentum=[0.5, 0.75, 0.8, 0.3, 0.5, 0.4, 0.2, 0.6, 0.45, 0.7],
                   branch=[0,0.05, 0.1, 0, 0, 0.1, 0.02, 0, 0, 0.25],
                   con_min_dist=0.8, verbose=True, safety=True, domain_size = 20, random_start=False)
data = x.generate_data(5000)
datax = data[:,0:-1]
max = np.max(datax) - np.min(datax)
for d in range(len(datax[0])):
    datax[:,d] = (datax[:,d] - np.min(datax[:,d]))/max
datay = data[:,-1]
```
The algorithm results for this setting can be found in the folder results/new_data_scaled_results. The .csv files contain the actual performance results for the algorithms. The best_labels.npy files contain the labels of the best-performing clustering for the respective seeds for each algorithm. The best performance was considered to be the highest sum of the NMI and ARI. In the case of ties, the earlier label set was kept.


The unused datasets of the 'low'-setting (datasets/low_data_{dim}.npy) were generated using the following code. The results for the various algorithms on the 'low' datasets can be found in results/low_data_results.
```
for dim in [2,5,10,50,100]:
    x = datagen.densityDataGen(dim=dim, ratio_noise = 0.1, max_retry=5, dens_factors=[1,1,0.5, 0.3, 2, 1.2, 0.9, 0.6, 1.4, 1.1], square=True, 
                      clunum= 10, seed = 6, core_num= 200, momentum=0, step=1,
                      branch=0, star=0, verbose=False, safety=False, domain_size = 20, random_start=False)
    data = x.generate_data(5000)
    datax = data[:,0:-1]
    max = np.max(datax) - np.min(datax)
    for d in range(len(datax[0])):
        datax[:,d] = (datax[:,d] - np.min(datax[:,d]))/max
    datay = data[:,-1]
```

## Seed Spreader ##

The seed_spreader code is based on the description provided in Junhao Gan and Yufei Tao. "DBSCAN revisited: Mis-claim, un-fixability, and approximation.", Proceedings of the 2015 ACM SIGMOD international conference on management of data. 2015, as well as Junhao Gan and Yufei Tao. "On the Hardness and Approximation of Euclidean DBSCAN", ACM Transactions on Database Systems (TODS), vol. 42, no. 3, pp. 1–45, 2017.
The var_density flag swaps the generator to the Varying-density setting described in the 2017 paper.

It can be used like this:

| **Parameter**        | **Abrev.** | **Default** | **Role**                                                                                                                                                                                                                                                                      |
|----------------------|-------------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| n      | n | 2000000 | number of data points   |
| dim      | d | 5 | dimensionality |
| ratio_noise    | $\rho$, $\rho_{noise}$ | 0.001 | noise ratio  |
| domain_size | | $10^5$ | domain size |
| reset_counter | $c_{reset}$ | 100 | local counter for hypersphere points |
| restart_chance_mult | $rcm$ | 0.001 | cluster restart overall<br>roughly the desired cluster number, though not the actual cluster number<br>corresponds to $\rho_{noise} =  rcm  / (n \cdot (1 - \rho_{noise}))$  |
| radius | $r_{vincinity}$ | 100 |  radius of hypersphere |
| noise_adapt | | False | adapt noise to altered domain size after random walk (not part of the description provided by Junhao Gan and Yufei Tao) |
| var_density| | False | enables density-based mode introduced in "On the Hardness and Approximation of Euclidean DBSCAN"|
| seed | | 0 | seed of randomness |
| verbose | | False | Generate text outputs at key events             |

The density variance is set to $radius \cdot ((i\ \textrm{mod}\ 10) + 1)$, which corresponds to the setting in the 2017 paper.

The step size ($r_{shift}$) is fixed to $radius\cdot \frac{dim}{2}$, which corresponds to both settings provided by Junhao Gan and Yufei Tao
```
data = seedSpreader(dim=2, noise_adapt=True, var_density = True)
```
