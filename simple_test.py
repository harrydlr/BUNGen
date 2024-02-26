from netgen import NetworkGenerator
import numpy as np
import random
from functools import reduce
import pickle
from numpy.random import default_rng
import warnings
import copy
import scipy
from scipy import stats
import logging

# Logs
logging.basicConfig(filename="myfile.txt", level=logging.DEBUG)
logging.captureWarnings(True)
#

# Target network
M_t, Pij_t, crows_t, ccols_t = NetworkGenerator.generate(30, 20, 3, bipartite=True, P=0.4, mu=0.2,
                                                         y_block_nodes_vec=[15, 10, 5], x_block_nodes_vec=[10, 5, 5],
                                                         fixedConn=True, link_density=0.2)

print("end")
