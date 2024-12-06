# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from dolfinx.fem import (Function, FunctionSpace, VectorFunctionSpace,
                         dirichletbc, locate_dofs_topological, Constant)
from dolfinx.mesh import exterior_facet_indices
from dolfinx.fem.petsc import (
    NonlinearProblem,
    assemble_vector,
    assemble_matrix,
    assemble_matrix_mat,
    create_vector,
    apply_lifting,
    set_bc,
)
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc

from ufl.algebra import Abs
from ufl import (lhs, rhs, action, derivative, TrialFunction)

def set_default_solver_parameters(prm):
    prm.setTolerances(rtol = 1e-12, atol=1e-14, divtol=None, max_it=5000)

from .petsc_solver import PETScSolver, LinearProblem
from .transient_nl_solver import TransientNLSolver
from .custom_nl_solver import CustomNLSolver
