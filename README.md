# DENSIRED

## Paarmeters
| **Parameter**        | **Abrev.** | **Default** | **Role**                                                                                                                                                                                                                                                                      |
|----------------------|------------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dim                  | $\di$      | 2           | dimensionality of the data set                                                                                                                                                                                                                                                |
| clunum               | \clunum    | 2           | number of clusters                                                                                                                                                                                                                                                            |
| core\_num            | \corenum   | 10          | number of cores across all clusters allows for explicit specification of core number per cluster                                                                                                                                                                              |
| domain\_size         | \domsize   | $1$         | initial range of the starting position of the clusters                                                                                                                                                                                                                        |
| radius               | \radius    | $1$         | base range around core center where points can be generated                                                                                                                                                                                                                   |
| shift                | \shift     | $1$         | base distance between cores                                                                                                                                                                                                                                                   |
| scale\_factors       | \scalef    | False       | cluster scale factors assigns all clusters a scale factor $cf_{i}$ of $1$ if False assigns random scale factors \sscalef$_{i}$ between $0.5$ and $2$ if True                                                                                                                  |
| stickiness           | \stick     | $0$         | cluster density factors assigns the value to all clusters as stickiness factor $cs_{i}$ assigns random stickiness factors $cs_{i}$ between $0$ and $1$ if True                                                                                                                |
| min\_dist            | \mindist   | $1.1$       | factor on the minimal distance between cores (sum of core sizes)                                                                                                                                                                                                              |
| shift\_normal\_width | \shinw     | 0           | width of normal distribution on shift                                                                                                                                                                                                                                         |
| max\_retry           | \maxretry  | 5           | maximal number of attempts for generation before generation enters the failure handling                                                                                                                                                                                       |
| branch               | \branch    | 0           | chance of creating a branch from the prior scaffolding of the cluster done by restarting from a randomly selected core of the current cluster                                                                                                                                 |
| star                 | \sta       | 0           | priority of the initial starting core for any attempt of restarting applies to both failure and branching                                                                                                                                                                     |
| seed                 | \seed      | $0$         | seed for random operations                                                                                                                                                                                                                                                    |
| connections          | \con       | $0$         | number of connections randomly picks the specified number of connection pairs allows for explicit specification of desired connections if a list is given connections have to specified as "startid;endid"  only guarantees number of attempted connections, not final number |
| con\_radius          | \conrad    | $2$         | analogous to radius, used for connections                                                                                                                                                                                                                                     |
| con\_shift           | \conshift  | $2$         | analogous to shift, used for connections                                                                                                                                                                                                                                      |
| con\_scale\_factors  | \conscalef | False       | analogous to scale\_factors, used for connections                                                                                                                                                                                                                             |
| con\_stickiness      | \constick  | $0.9$       | analogous to stickiness, used for connections                                                                                                                                                                                                                                 |
| con\_min\_dist       | \conmind   | $0.9$       | factor on the minimal distance between a connection and other cores                                                                                                                                                                                                           |
| data number          | \num       |             | number of data points (parameter of data generation function, unlike other parameters not set for generator instance)                                                                                                                                                         |
| clu\_ratios          | \clurat    | None        | distribution of data points across clusters                                                                                                                                                                                                                                   |
| min\_ratio           | \minrat    | 0           | minimal ratio of cores/points (only used when these are randomly determined through $csr$, should be $<<1$)                                                                                                                                                                   |
| ratio\_noise         | \norat     | $0$         | ratio of noise data points                                                                                                                                                                                                                                                    |
| ratio\_con           | \conrat    | $0$         | ratio of connection data points                                                                                                                                                                                                                                               |
| square               | \squ       | False       | generate noise in square data space                                                                                                                                                                                                                                           |
| verbose              | \ver       | False       | Generate text outputs at key events                                                                                                                                                                                                                                           |
| safety               | \saf       | True        | activates various safety features, such as: noise generation only outside of two times core radius guarantee of at least a single point per core for data generation                                                                                                          |



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
