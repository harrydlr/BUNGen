# BUNGen: Synthetic generator for structured ecological networks

BUNGen (Bipartite and Unipartite Network Generator) is a Python software designed to address the methodological gap in synthetic network generation. BUNGen facilitates the creation of uni- and bipartite networks with varying levels of prescribed structures, offering a versatile tool for exploring the consequences of network architecture in ecological studies.
For more details, refer to the paper #LinkToPaper#

## Project Overview

### Project Structure

    .
    ├── config                                # Path config files
    │   ├── __init__.py
    │   └── config.py                       
    ├── Empirical_Net
    │   ├── Pollination
    │   └── Seed_Dispersal
    ├── figures                              # Contains the BUNGen paper figures
    ├── netgen                               # Main library
    │   ├── NetworkGenerator.py              # Generate structured networks
    │   ├── generate_probability_matrix.py   # Generate synthetic network matrix of link probabilities with the define parameters
    │   ├── utils.py                         # Util functions used in NetworkGenerator
    │   └── __init__.py
    ├── scripts                              # 
    │   ├── __init__.py
    │   └── paper_figures_bungen.py          # Produce the BUNGen paper figures
    ├── tests                                # Unitary tests
    │   ├── __init__.py
    │   └── bungen_test.py
    ├── .gitignore
    ├── README.md
    ├── LICENSE
    └── requirements.txt                    # Python software requirements

## Getting Started

To install and use BUNGen, follow the steps below

### Prerequisites

    - Python 3.8+
    - Pip

### Installation

Clone the repository:

    git clone https://github.com/COSIN3-UOC/BUNGen.git


Install the required packages:

    pip install -r requirements.txt


## Usage
| Parameter                  | Type                                                                                              | Description                                                                                                                              |
|-----------------------------|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `rows`                      | `int`                                                                                             | Number of row nodes.                                                                                                                     |
| `cols`                      | `int`                                                                                             | Number of column nodes.                                                                                                                  |
| `block_number`              | `int` $\geq 1$                                                                                    | Number of prescribed blocks in the network.                                                                                              |
| `p`                         | `float` $\in [0, 1]`, or `list` $\in [0,1]^{block\_number}$                                       | Noise outside a perfectly nested structure. If `p` is a `list` of length `block_number`, `p[${\alpha}$]` indicates the amount of this noise in block $\alpha$. |
| `mu`                        | `float` $\in [0, 1]$                                                                              | Inter-block (i.e., between-modules) noise.                                                                                               |
| `y_block_nodes_vec`         | `list` $` \in \mathbb{N}^{block\_number} `$ \| $$`\sum_{i=1}^{block\_number} x_i`$$ = $`\texttt{cols}`$ | Number of nodes per block in the y-axis                                                                                   |
| `x_block_nodes_vec`         | `list` $\in \mathbb{N}^{block_number}$ $\mid \sum{i=1}^block_number x_i = \texttt{cols}$          | Number of nodes per block in the x-axis.                                                                                                 |
| `bipartite`                 | `boolean`                                                                                         | `True` for bipartite networks, `False` for unipartite (default).                                                                         |
| `fixedConn`                 | `boolean`                                                                                         | `True`: to produce a network with prescribed connetance. `False`: to set a specific $\xi$ value.                                         |
| `link_density`              | `float`                                                                                           | If `fixedConn` = `True`, it specifies the desired connectance $\in [0,1]$. If `fixedConn` $=$ `False`, it specifies $\xi > 0$.           |

## Inputs:
| Parameter               | Type                                                                                        | Description                                                                                                                                                  |
|--------------------------|---------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `rows`                   | `int`                                                                                       | Number of row nodes.                                                                                                                                         |
| `cols`                   | `int`                                                                                       | Number of column nodes.                                                                                                                                      |
| `block_number`           | `int` $\geq 1$                                                                              | Number of prescribed blocks in the network.                                                                                                                  |
| `p`                      | `float` $\in [0, 1]$, or `list` $\in [0,1]^{block\_number}$                                 | Noise outside a perfectly nested structure. If `p` is a `list` of length `block\_number`, `p[${\alpha}$]` indicates the amount of this noise in block $\alpha$. |
| `mu`                     | `float` $\in [0, 1]$                                                                        | Inter-block (i.e., between-modules) noise.                                                                                                                   |
| `y_block_nodes_vec` | `list` $ \in \mathbb{N}^{block\_number}$ \| $\sum_{i=1}^{block\_number} x_i = \texttt{rows}$ | Number of nodes per block in the y-axis. |
| `x_block_nodes_vec`      | `list` $\in \mathbb{N}^{block\_number}$ $\mid \sum_{i=1}^{block\_number} x_i = \texttt{cols}$ | Number of nodes per block in the x-axis.                                   |
| `bipartite`              | `boolean`                                                                                   | `True` for bipartite networks, `False` for unipartite (default).                                                                                              |
| `fixedConn`              | `boolean`                                                                                   | `True`: to produce a network with prescribed connectance. `False`: to set a specific $\xi$ value.                                                          |
| `link_density`           | `float`                                                                                     | If `fixedConn` $=$ `True`, specifies the desired connectance $\in [0,1]$. If `fixedConn` $=$ `False`, it specifies $\xi > 0$.                           |


positional arguments:
1) rows  = int: number of row nodes.
2) cols  = int: number of column nodes.
3) block_number   = int >= 1: number of prescribed blocks (i.e. modules) in the network.
4) P   = in [0, 1] parameter that controls the amount of noise outside a perfectly nested structure.
5) mu  = in [0, 1] parameter that controls the amount of inter-block (i.e. between modules) noise.
6) gamma = float: bounded in (x1,x2), gamma is the scaling parameter of the distribution of block sizes. Default is gamma = 2.5. 
	For networks with equally-sized blocks set gamma = 0.
7) bipartite = bool: True for bipartite networks, false for unipartite. If not given, it will generate a unipartite network.
8) min_block_size = int: minimum block size, default 10% of rows and cols, respectively.
9) fixedConn = bool: True if you want to produce a network with prescribed connetance. False if you want to set a specific xi value.
10) link_density = float: If fixedConn = True, this parameter specifies the desired connectance [0,1]. If fixedConn = False, it specifies xi >= 0.

## Output:
A numpy matrix that corresponds to the binary synthetic adjacency matrix (biadjacency for bipartite cases), and/or a numpy matrix with link probabilities, two lists of ints containing the rows and columns partition labels, respectively.

## Use examples: 
### To use as a library
To produce a single network with desired parameters within a custom made script. User can proceed in the following way.

```python
from netgen import NetworkGenerator

M, *_ = NetworkGenerator.generate(500, 500, 4, bipartite=True, p=0.5, mu=0.5,
								  gamma=2.5, min_block_size=0, fixedConn=False, link_density=2.45)

```
Keep in mind that the parameters are positional. If user does not pass the parameters as named arguments, then order must be respected. If the user wants the function to return the matrix of link probabilities edit the line above by replacing M,* _ = with M,Pij,* _ =  or if the users wants the rows and columns partition labels edit the line above as M,Pij,rowsLabels,colsLabels = 

To produce several networks of the same size and same number of blocks, while varying some parameter and keeping others fixed:

```python
from netgen import NetworkGenerator

gen = NetworkGenerator(500, 500, 4, bipartite=True, p=0.5, mu=0.5, gamma=2.5,
                       min_block_size=0, fixedConn=False, link_density=2.45)

for p in np.arange(0, 1, 0.2):
    M, *_ = gen(P=p)
    # do something with each M (plot, save, append, etc)

```

### From the commmand line
This script will save the output matrices in csv files. Produce a network with certain number of rows, cols, blocks, xi and noise. (fixedConn false by default).
``` sh
python generate_synthetic_networks.py 100 200 2 0.1 0.1 2.05

```
If you want to produce a network with a given connectance replace fixedConn to true and change xi value for the desired connectance value by typing:
``` sh
python generate_synthetic_networks.py 100 200 2 0.1 0.1 .005 -f

```
To modify the remaining parameters just add -a value if you want to modify the gamma default value from the powerlaw
``` sh
python generate_synthetic_networks.py 100 200 2 0.1 0.1 .005 -f -ga 2.1

```

For help and description of the parameters type:
``` sh
python generate_synthetic_networks.py -h

```

# Citations
Harry R. de los Ríos, María J. Palazzi, Aniello Lampo, Albert Solé-Ribalta, Javier Borge-Holthoefer

A. Solé-Ribalta, CJ. Tessone, M S. Mariani, and J Borge-Holthoefer. Revealing in-block nestedness: Detection and benchmarking, Phys. Rev. E 97, 062302 (2018). DOI: [10.1103/PhysRevE.97.062302](https://doi.org/10.1103/PhysRevE.97.062302)

MJ Palazzi, J Borge-Holthoefer, CJ Tessone and A Solé-Ribalta. Macro- and mesoscale pattern interdependencies in complex networks. J. R. Soc. Interface, 16, 159, 20190553 (2019). DOI: [10.1098/rsif.2019.0553](https://doi.org/10.1098/rsif.2019.0553)
