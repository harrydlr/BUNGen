#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 21:44:01 2022

@author: mariapalazzi
"""
from numpy.random import rand, uniform
from typing import List
from math import ceil
import warnings
from scipy import stats


def heterogenousBlockSizes(ax: str, B: int, N: int, block_nodes_vec: list = []) -> List[int]:
    """
    This function will generate heterogenous block sizes, following a powerlaw distribution.
    As in Clauset A, et al. SIAM Rev., 51(4), 661–703 2009.

    The transformation method is used to generate x samples from a powerlaw distribution.
    These samples will be the sizes of the blocks across each dimension, the functions is called
    for col and rows, separately.
    Two elements are needed: uniformly distributed random source r where 0 <= r < 1 and
    the functional inverse of the cumulative density function (CDF) of the power-law distribution

    The source r is simply given as a parameter to the CDF.

    the inputs are:

    N: int
        numnber of nodes to split into block
    B: int
        number of blocks on which the nodes will be divided
    block_nodes_vec: list of ints
        Integers that represent the nodes per block
    output: list of ints
        a list containing numbers determining the size of each block
    """
    # Check Block number is right
    #if not isinstance(B, int) or B < 1:
    #    raise Exception("Block number should be an integer equal or greater than 1")
    # Check Rows and Cols are integers
    if not isinstance(N, int):
        raise Exception(f'Rows and columns must be integers')
    if len(block_nodes_vec) == 0:
        if N % B != 0:
            warnings.warn(f"The number of nodes is not divisible by B. {N} % {B} = {N%B}")
            warnings.warn(f"The remaining {N%B} node(s) will be redistributed along the blocks.")

        return [ N//B + (ceil((N%B)/B) if (N%B-(b)>0) else 0) for b in range(B)]

    else:
        if not all(isinstance(n, int) for n in block_nodes_vec):
            raise Exception(f'{ax}_block_nodes_vec list must contain integers')
        if sum(block_nodes_vec) != N:
            raise Exception(f'The sum of elements in {ax}_block_nodes_vec must be {N} but '
                            f'it is {sum(block_nodes_vec)}')
        if len(block_nodes_vec) != B:
            raise Exception(f'The length of {ax}_block_nodes_vec list must be {B} but '
                            f'it is {len(block_nodes_vec)}')
        #colsizes = block_nodes_vec.sort(reverse=True)
        #return colsizes
        return block_nodes_vec
