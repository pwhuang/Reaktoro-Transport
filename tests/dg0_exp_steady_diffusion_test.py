# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os

os.environ["OMP_NUM_THREADS"] = "1"

from retropy.problem import TracerTransportProblemExp
from retropy.physics import DG0Kernel
from retropy.solver import TransientNLSolver
from retropy.manager import XDMFManager
from retropy.benchmarks import DiffusionBenchmark

from dolfinx.fem import Constant
from utility_functions import convergence_rate
from math import isclose


class DG0ExpSteadyDiffusionTest(
    TracerTransportProblemExp,
    DiffusionBenchmark,
    DG0Kernel,
    TransientNLSolver,
    XDMFManager,
):
    def __init__(self, nx, is_output):
        marked_mesh = self.get_mesh_and_markers(nx, "triangle")
        super().__init__(marked_mesh)

        self.set_flow_field()
        self.define_problem()

        self.set_problem_bc()

        self.generate_solver()
        self.set_solver_parameters(linear_solver="cg", preconditioner="jacobi")

        if is_output == True:
            self.generate_output_instance("steady_diffusion_exp")

    def set_problem_bc(self):
        values = super().set_problem_bc()
        # When solving steady-state problems, the diffusivity of the diffusion
        # boundary is a penalty term to the variational form.
        self.add_component_diffusion_bc(
            "solute", diffusivity=Constant(self.mesh, 1e2), values=values
        )


list_of_nx = [10, 20]
element_diameters = []
err_norms = []

for i, nx in enumerate(list_of_nx):
    problem = DG0ExpSteadyDiffusionTest(nx, is_output=False)
    problem.solve_transport()
    problem.get_solution()
    error_norm = problem.get_error_norm()
    err_norms.append(error_norm)
    element_diameters.append(problem.get_mesh_characterisitic_length())

print(err_norms)

convergence_rate_m = convergence_rate(err_norms, element_diameters)
print(convergence_rate_m)


def test_function():
    assert isclose(convergence_rate_m[0], 1, rel_tol=0.2)
