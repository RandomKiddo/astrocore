"""
This code is for linear algebra functions on astrophysical systems (but can also be used generally).

Currently, this subpackage implements:
- LU Decomposition using Doolittle Method.

* Copyright © 2025 RandomKiddo
"""

import numpy as np


from typing import *


def lu_decomposition(mat: Union[list[list[float]], list[list[int]], np.array], n: int = -1) -> Tuple[np.array, np.array]:
    """
    Performs LU decomposition on a given matrix of size, using Doolittle's Method. <br>
    :param mat: The matrix to do decomposition on, a list of lists or numpy array. <br>
    :param n: The size for the decomposition. Defaults to the row length of the matrix. <br>
    :return: The lower and upper matrices. 
    """

    # Set the length.
    if n < 0:
        n = len(mat)

    # Initialize the upper and lower triangular matrices.
    lower = np.zeros((n, n))
    upper = np.zeros((n, n))
    
    # Start decomposing.
    for i in range(n):
        # Upper triangular matrix.
        for k in range(i, n):
            # Summation of lower at (i, j) times upper at (j, k).
            sum = 0
            for j in range(i):
                sum += (lower[i][j]*upper[j][k])
            
            # Evaluate upper at (i, k).
            upper[i][k] = mat[i][k]-sum

        # Lower triangular matrix.
        for k in range(i, n):
            if i == k:
                lower[i][i] = 1 # Diagonal is 1
            else:
                # Summation of L at (k, j) times upper at (j, i).
                sum = 0
                for j in range(i):
                    sum += (lower[k][j]*upper[j][i])
                
                # Evaluate lower at (k, i).
                lower[k][i] = int((mat[k][i]-sum) / upper[i][i])
    
    # Return the lower then upper matrices as numpy arrays.
    return lower, upper 


def cholesky_decomposition():
    pass


def cg_method():
    pass


def gmres_method():
    pass


def preconditioners():
    pass

