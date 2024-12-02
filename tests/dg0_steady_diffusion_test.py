# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os

os.environ["OMP_NUM_THREADS"] = "1"

from retropy.physics import DG0Kernel
from retropy.solver import PETScSolver
from retropy.benchmarks import DiffusionBenchmark

from utility_functions import convergence_rate

from dolfinx.fem import Constant
from math import isclose


class DG0SteadyDiffusionTest(DiffusionBenchmark, DG0Kernel, PETScSolver):
    def __init__(self, nx):
        marked_mesh = self.get_mesh_and_markers(nx, "triangle")
        super().__init__(marked_mesh)

        self.set_flow_field()
        self.define_problem()
        self.set_problem_bc()

        self.generate_solver()
        self.set_solver_parameters(linear_solver="gmres", preconditioner="jacobi")

    def set_problem_bc(self):
        values = DiffusionBenchmark.set_problem_bc(self)
        # When solving steady-state problems, the diffusivity of the diffusion
        # boundary is a penalty term to the variational form.
        self.add_component_diffusion_bc(
            "solute",
            diffusivity=Constant(self.mesh, 1e2),
            kappa=Constant(self.mesh, 1.0),
            values=values,
        )


list_of_nx = [10, 20]
element_diameters = []
err_norms = []

for nx in list_of_nx:
    problem = DG0SteadyDiffusionTest(nx)
    problem.solve_transport()
    numerical_solution = problem.get_solution()
    error_norm = problem.get_error_norm()
    err_norms.append(error_norm)
    element_diameters.append(problem.get_mesh_characterisitic_length())

print(err_norms)

convergence_rate_m = convergence_rate(err_norms, element_diameters)
print(convergence_rate_m)


def test_function():
    assert isclose(convergence_rate_m[0], 1, rel_tol=0.2)
