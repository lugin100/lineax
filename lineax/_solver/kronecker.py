# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, TypeAlias

import jax
from jaxtyping import Array, PyTree

from .._operator import AbstractLinearOperator, KroneckerLinearOperator
from .._solution import RESULTS
from .._solve import _SolverState, AbstractLinearSolver, AutoLinearSolver


_KroneckerState: TypeAlias = tuple[
    int, int, AbstractLinearSolver, AbstractLinearSolver, _SolverState, _SolverState
]


class Kronecker(AbstractLinearSolver[_KroneckerState]):
    r"""Solver for linear systems with a KroneckerLinearOperator.

    The Kronecker structure is exploited to bring the system $(A \otimes B)x = b$
    into the form $\text{mat}(b) = B\text{mat}(x)A^T$ via the vec-trick,
    where $\text{mat}$ is the reshaping operation of a $nm$-vector into a $n,m$-matrix.

    This system is solved by solving systems on the operators $A$ and $B$.
    AutoLinearSolver is used to determine the best solvers for $A$ and $B$,
    and passed options are lowered down.
    """

    def init(
        self,
        operator: AbstractLinearOperator,
        options: dict[str, Any] = {},
    ) -> _KroneckerState:
        if not isinstance(operator, KroneckerLinearOperator):
            raise ValueError(
                "This solver is only suitable for KroneckerLinearOperators."
            )
        autoLinearSolver = AutoLinearSolver(well_posed=True)
        op1 = operator.operator1
        op2 = operator.operator2
        solver1 = autoLinearSolver.select_solver(op1)
        solver2 = autoLinearSolver.select_solver(op2)
        solver1_state = solver1.init(op1, options)
        solver2_state = solver2.init(op2, options)
        m = op1.in_structure().shape[0]
        n = op2.in_structure().shape[0]
        return (m, n, solver1, solver2, solver1_state, solver2_state)

    def compute(
        self, state: _KroneckerState, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        m, n, solver1, solver2, solver1_state, solver2_state = state
        del state

        B = vector.reshape((n, m), order="F")
        Z = jax.vmap(
            lambda b: solver2.compute(solver2_state, b, options)[0],
            in_axes=1,
            out_axes=1,
        )(B)
        XT = jax.vmap(
            lambda z: solver1.compute(solver1_state, z, options)[0],
            in_axes=1,
            out_axes=1,
        )(Z.mT)
        X = XT.mT
        x = X.reshape((-1,), order="F")
        return x, RESULTS.successful, {}

    def transpose(self, state: _KroneckerState, options: dict[str, Any]):
        raise NotImplementedError

    def conj(self, state: _KroneckerState, options: dict[str, Any]):
        raise NotImplementedError

    def assume_full_rank(self):
        return False
