# netgen/utils.py

from numpy import power, linspace
from scipy.interpolate import interp1d
from typing import List
from numpy.typing import ArrayLike


def ballcurve(x: ArrayLike, xi: float) -> ArrayLike:
    """
    function to generate the curve for the nested structure, given a shape
    parameter xi. If xi= 1 is linear.
    input:
    ----------
    x: 1D array, [0,1]
        initial values to be evaluated on the function
    xi: number, >=1
        shape parameter of how stylised is the curve
    output:
    ----------
    y: 1D array, [0,1]
        evaluated function
    """
    return 1 - power(1 - power(x, 1 / xi), xi)


def xiConnRelationship(rw: int, cl: int, xi: float) -> int:
    """
     Thi function calculates the connectance of a matrix (assuming B=1) for a given
     nested profile given by the shape parameter xi

     inputs:
    cl: (int)
         column number;
     M: (int)
         row number;
     xi: (float)
         nested profile;

     OUTPUT
     C: (float)
         matrix connectance;
    """
    E = 0  # edge counter
    for i in range(cl):
        x = i / cl  # tessellate
        y = ballcurve(x, xi)
        for j in range(rw):
            if j / rw >= y:
                E += 1
    return E


def xiFunConn(
    rowsList: List[int], colsList: List[int], rowTot: int, colTot: int, C: float
) -> float:
    """
    This function estimates the xi value for a matrix of given size, connectance
    and distribution of blocks sizes

    inputs:
    rowsList: list of int,
        containing the size of each block on rows dimension
    colsList: list of int,
        containing the size of each block on cols dimension
    rowTot: int,
        total number of rows nodes
    colTot: int,
        total number of cols nodes
    C: float
        the global connectance;

    output:
     xi: float
         the matrix nestedness perfile;
    """

    links = C * rowTot * colTot
    xiList = linspace(0.001, 5, 100)
    edgeList = []
    for xi in xiList:
        E = 0.0  # edge counter
        for i in range(len(rowsList)):
            blockRow = rowsList[i]
            blockCol = colsList[i]
            edgeBlock = xiConnRelationship(blockRow, blockCol, xi)
            E += edgeBlock
        edgeList.append(E)

    f = interp1d(edgeList, xiList)

    if links < min(edgeList):
        links = min(edgeList)
    elif links > max(edgeList):
        links = max(edgeList)

    xi = f([links])[0]
    return xi
