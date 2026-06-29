import abc

from jax import numpy as jnp

from ._operator import AbstractLinearOperator


class CompositeLinearOperator(AbstractLinearOperator):
    """Abstract base class for composite linear operators."""

    @abc.abstractmethod
    def solve(self, b):
        """Defines how a linear system for the given composite operator can be solved
        using linear solves on the constituent operators.
        """


class BlockLinearOperator(CompositeLinearOperator):
    """A Linear Operator defined as a 2x2 block of linear operators."""

    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.n, self.m, self.p, self.q = self.check_sizes()

    def check_sizes(self):
        n, m = self.A.out_size(), self.A.in_size()
        p, q = self.D.out_size(), self.D.in_size()
        assert self.B.out_size == n
        assert self.B.in_size == q
        assert self.C.in_size == m
        assert self.C.out_size == p
        return n, m, p, q

    def mv(self, vector):
        return (
            self.A.mv(vector[0]) + self.B.mv(vector[1]),
            self.C.mv(vector[0]) + self.D.mv(vector[1]),
        )

    def as_matrix(self):
        return jnp.block(
            [
                [self.A.as_matrix(), self.B.as_matrix()],
                [self.C.as_matrix(), self.D.as_matrix()],
            ]
        )

    def transpose(self):
        return BlockLinearOperator(
            self.A.transpose(),
            self.C.transpose(),
            self.B.transpose(),
            self.D.transpose(),
        )

    def in_structure(self):
        return (self.A.in_structure(), self.D.in_structure())

    def out_structure(self):
        return (self.A.out_structure(), self.D.out_structure())

    def solve(self, b):
        # Solve via Schur complement,
        # exploit possible triangularity
        raise NotImplementedError
