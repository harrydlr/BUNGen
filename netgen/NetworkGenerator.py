from dataclasses import dataclass
from typing import List, Tuple
from numpy.typing import ArrayLike
from numpy import array, fill_diagonal, triu, repeat
from .network_generator import network_generator
from .heterogenousBlockSizes import heterogenousBlockSizes
from .xiFunConn import xiFunConn
from numpy.random import uniform
import warnings


@dataclass
class NetworkGenerator:
    rows: int
    columns: int
    block_number: int
    P: list
    mu: float
    y_block_nodes_vec: list
    x_block_nodes_vec: list
    bipartite: bool
    fixedConn: bool
    link_density: float

    def get_block_sizes(self) -> Tuple[List[int], List[int]]:
        # Check Block number is integer and greater than 1
        if isinstance(self.block_number, int):
            if self.block_number < 1:
                raise Exception("Block number should be an integer equal or greater than 1")
        else:
            raise Exception("Block number should be an integer")
        # Check rows and cols are int
        if not isinstance(self.rows, int):
            raise Exception(f'Rows and columns must be integers')
        if not isinstance(self.columns, int):
            raise Exception(f'Rows and columns must be integers')

        self.cy = heterogenousBlockSizes(
            "y", self.block_number, self.rows, self.y_block_nodes_vec)
        #if self.rows == self.columns:
        #    self.cx = self.cy
        #else:
        self.cx = heterogenousBlockSizes(
            "x", self.block_number, self.columns, self.x_block_nodes_vec)
        return self.cx, self.cy

    def synthetic_network(self) -> Tuple[ArrayLike, ArrayLike, List[int], List[int]]:
        Mij = network_generator(self.rows, self.columns, self.block_number, self.cy, self.cx, self.xi, self.P, self.mu)
        Mrand = array(uniform(0, 1, size=(self.rows, self.columns)))
        labelRows = repeat(range(len(self.cy)),self.cy).tolist()
        labelCols = repeat(range(len(self.cx)),self.cx).tolist()
        M = (Mij > Mrand).astype(int)
        if not self.bipartite:
            fill_diagonal(M, 0)
            M = triu(M, k=1) + (triu(M, k=1)).T
        return M, Mij, labelRows, labelCols

    @property
    def xi(self) -> float:
        if self.fixedConn == True:
            maxConn = sum([(x*y) for x,y in zip(self.cx,self.cy)])/(self.rows*self.columns)
            xi = xiFunConn(self.cy, self.cx, self.rows, self.columns, self.link_density)
            if maxConn < self.link_density:
                raise ValueError(f"Desired connectance not possible for parameters combination. Max connectance {maxConn:.3f}")
            else:
                print(f"xi value for desired connectance {xi:.2f}") 
        else:
            xi = round(self.link_density, 2)
        return xi
    
    @property
    def net_type(self) -> str:
        return "bipartite" if self.bipartite else "unipartite"
    
    def __post_init__(self) -> None:
        self.get_block_sizes()
        if self.fixedConn and self.link_density>1:
            raise ValueError("If parameter 'fixedConn' is True, then 'link_density' cannot be greater than 1")
    
    def __call__(self, **kwargs) -> Tuple[ArrayLike, ArrayLike, List[int], List[int]]:
        for param in self.__annotations__.keys():
            if param in kwargs:
                setattr(self, param, kwargs[param])

        # Check Block number is integer and greater than 1
        if isinstance(self.block_number, int):
            if self.block_number < 1:
                raise Exception("Block number should be an integer equal or greater than 1")
        else:
            raise Exception("Block number should be an integer")
        # Check P is right
        # Check P is float/int or list
        if not isinstance(self.P, (float, list)):
            raise Exception("P must be a float or a list of floats")
        # If P is float, create a list of length block_number with P as values
        if isinstance(self.P, float):
            # Check that P is in range
            if float(0) > self.P or self.P > float(1):
                raise Exception("P values must be in range [0,1]")
            # Create list of P values
            self.P = [self.P for i in range(self.block_number)]
        # If P is a list
        else:
            # Check that list length is equal to number of blocks
            if not len(self.P)==self.block_number:
                raise Exception(f"List P must have a length of {self.block_number} "
                                f"but it has a length of {len(self.P)}")
            # Check that all values of list P are float
            if not all(isinstance(n, float) for n in self.P):
                raise Exception("List P must contain only float")
            # Check that list P values are in range
            if not all(n >= float(0) and n <= float(1) for n in self.P):
                raise Exception("P values must be in range [0,1]")
        # Round P values
        self.P = [round(num, 2) for num in self.P]

        # Check mu is flot and is in range
        if isinstance(self.mu, float):
            if float(0) > self.mu or self.mu > float(1):
                raise Exception("mu value must be in range [0,1]")
        else:
            raise Exception("mu must be a float")
        # Round mu
        self.mu = round(self.mu, 2)
        # Check bipartite param
        if not isinstance(self.bipartite, bool):
            raise Exception("Bipartite parameter may be boolean")
        # Unipartite check
        if not self.bipartite and self.columns != self.rows:
            raise ValueError("For unipartite configuration, the number of columns and rows must be the same.")

        # Check FixedConn param
        if not isinstance(self.fixedConn, bool):
            raise Exception("fixedConn parameter may be boolean")

        # Check mu is flot and is in range
        if isinstance(self.link_density, float):
            if float(0) > self.link_density or self.link_density > float(1):
                raise Exception("link_density parameter may be in range [0,1]")
        else:
            raise Exception("link_density may be a float in range [0,1]")

        return self.synthetic_network()

    @classmethod
    def generate(
        cls,
        rows: int,
        columns: int,
        block_number: int,
        P: list,
        mu: float,
        y_block_nodes_vec: list,
        x_block_nodes_vec:list,
        bipartite: bool,
        fixedConn: bool,
        link_density: float,
    ):
        return cls(
            rows=rows,
            columns=columns,
            block_number=block_number,
            P=P,
            mu=mu,
            y_block_nodes_vec=y_block_nodes_vec,
            x_block_nodes_vec=x_block_nodes_vec,
            bipartite=bipartite,
            fixedConn=fixedConn,
            link_density=link_density,
        )()
