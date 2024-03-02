from dataclasses import dataclass
from typing import List, Tuple
from numpy.typing import ArrayLike
from numpy import array, fill_diagonal, triu, repeat
from math import ceil
from .generate_probability_matrix import generate_probability_matrix
from .utils import xiFunConn
from numpy.random import uniform
import warnings


@dataclass
class NetworkGenerator:
    rows: int
    columns: int
    block_number: int
    P: List[float]
    mu: float
    y_block_nodes_vec: List[int]
    x_block_nodes_vec: List[int]
    bipartite: bool
    fixedConn: bool
    link_density: float

    def get_block_sizes(self) -> Tuple[List[int], List[int]]:
        self._validate_block_number()
        self._validate_rows_and_columns()
        #self._validate_block_nodes_vec("y", self.block_number, self.rows, self.y_block_nodes_vec)
        #self._validate_block_nodes_vec("x", self.block_number, self.columns, self.x_block_nodes_vec)
        #self.cy = self.y_block_nodes_vec
        #self.cx = self.x_block_nodes_vec
        self.cy = self._validate_block_nodes_vec("y", self.block_number, self.rows, self.y_block_nodes_vec)
        self.cx = self._validate_block_nodes_vec("x", self.block_number, self.columns, self.x_block_nodes_vec)
        return self.cx, self.cy

    def synthetic_network(self) -> Tuple[ArrayLike, ArrayLike, List[int], List[int]]:
        Mij = generate_probability_matrix(self.rows, self.columns, self.block_number, self.cy, self.cx, self.xi, self.P, self.mu)
        Mrand = array(uniform(0, 1, size=(self.rows, self.columns)))
        labelRows = repeat(range(len(self.cy)), self.cy).tolist()
        labelCols = repeat(range(len(self.cx)), self.cx).tolist()
        M = (Mij > Mrand).astype(int)
        if not self.bipartite:
            fill_diagonal(M, 0)
            M = triu(M, k=1) + (triu(M, k=1)).T
        return M, Mij, labelRows, labelCols

    @property
    def xi(self) -> float:
        if self.fixedConn:
            max_conn = sum([(x * y) for x, y in zip(self.cx, self.cy)]) / (self.rows * self.columns)
            xi = xiFunConn(self.cy, self.cx, self.rows, self.columns, self.link_density)
            self._validate_max_conn(max_conn, self.link_density)
            print(f"xi value for desired connectance: {xi:.2f}")
        else:
            #xi = round(self.link_density, 2)
            xi = self.link_density
        return xi

    @property
    def net_type(self) -> str:
        return "bipartite" if self.bipartite else "unipartite"

    def __post_init__(self) -> None:
        self.get_block_sizes()
        self._validate_link_density()
        self._validate_P()
        self._validate_mu()
        self._validate_bipartite()

    def __call__(self, **kwargs) -> Tuple[ArrayLike, ArrayLike, List[int], List[int]]:
        self._update_params(kwargs)
        self.get_block_sizes()
        self._validate_block_number()
        return self.synthetic_network()

    @classmethod
    def generate(
            cls,
            rows: int,
            columns: int,
            block_number: int,
            P: List[float],
            mu: float,
            y_block_nodes_vec: List[int],
            x_block_nodes_vec: List[int],
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

    # Private methods
    def _validate_block_number(self) -> None:
        if not isinstance(self.block_number, int) or self.block_number < 1:
            raise ValueError("Block number should be an integer greater than or equal to 1")

    def _validate_rows_and_columns(self) -> None:
        if not isinstance(self.rows, int) or not isinstance(self.columns, int):
            raise ValueError("Rows and columns must be integers")

    def validate_single_P(self, P: float) -> List[float]:
        if not (0 <= P <= 1):
            raise ValueError("P value must be in range [0, 1]")
        return [P] * self.block_number

    def validate_list_P(self, P: List[float]) -> List[float]:
        if len(P) != self.block_number:
            raise ValueError(f"List P must have a length of {self.block_number}, but it has a length of {len(P)}")
        if not all(isinstance(n, float) for n in P):
            raise ValueError("List P must contain only floats")
        if not all(0 <= n <= 1 for n in P):
            raise ValueError("P values must be in range [0, 1]")
        #return [round(num, 2) for num in P]
        return P

    def _validate_P(self) -> None:
        if isinstance(self.P, float):
            self.P = self.validate_single_P(self.P)
        elif isinstance(self.P, list):
            self.P = self.validate_list_P(self.P)
        else:
            raise ValueError("P must be a float or a list of floats")

    def _validate_bipartite(self) -> None:
        if not isinstance(self.bipartite, bool):
            raise ValueError("Bipartite parameter must be a boolean")
        if not self.bipartite and self.columns != self.rows:
            raise ValueError("For unipartite configuration, the number of columns and rows must be the same.")

    def _validate_mu(self) -> None:
        if not isinstance(self.mu, float) or not (0 <= self.mu <= 1):
            raise ValueError("mu value must be a float in range [0,1]")

    def _validate_max_conn(self, max_conn: float, link_density: float) -> None:
        if max_conn < link_density:
            raise ValueError(f"Desired connectance not possible. Max connectance: {max_conn:.3f}")

    def _validate_link_density(self) -> None:
        if isinstance(self.link_density, (int, float)):
            if self.fixedConn:
                if not (0 <= self.link_density <= 1):
                    raise ValueError("If 'fixedConn' is True, 'link_density' should be in the range [0, 1]")
            elif self.link_density <= 0:
                raise ValueError("If 'fixedConn' is False, 'link_density' should be greater than 0")
        else:
            raise Exception("link_density may be a float")

    def _validate_block_nodes_vec(self, ax: str, B: int, N: int, block_nodes_vec: list) -> None:
        """
        Validate block_nodes_vec for x or y dimension.

        Parameters:
            ax (str): Dimension identifier ('x' or 'y').
            B (int): Number of blocks.
            N (int): Number of nodes.
            block_nodes_vec (list): List of integers representing nodes per block.

        Raises:
            ValueError: If the validation fails.
        """
        if B == 1 and not block_nodes_vec:
            # Automatically assume all nodes for a single block
            return [N]
        if not block_nodes_vec:
            # Assume equally sized blocks
            if N % B != 0:
                warnings.warn(f"The number of nodes is not divisible by B. {N} % {B} = {N % B}")
                warnings.warn(f"The remaining {N % B} node(s) will be redistributed along the blocks.")
            return [N // B + (ceil((N % B) / B) if N % B - b > 0 else 0) for b in range(B)]

        if not all(isinstance(n, int) for n in block_nodes_vec):
            raise ValueError(f'{ax}_block_nodes_vec list must contain integers')
        if len(block_nodes_vec) != B:
            raise ValueError(f'The length of {ax}_block_nodes_vec list must be {B} but '
                             f'it is {len(block_nodes_vec)}')
        if sum(block_nodes_vec) != N:
            raise ValueError(f'The sum of elements in {ax}_block_nodes_vec must be {N} but '
                             f'it is {sum(block_nodes_vec)}')
        return block_nodes_vec

    def _update_params(self, kwargs) -> None:
        for param in self.__annotations__.keys():
            if param in kwargs:
                setattr(self, param, kwargs[param])
