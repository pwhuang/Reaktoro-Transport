# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os

os.environ["OMP_NUM_THREADS"] = "1"

from retropy.physics import DG0Kernel
from retropy.solver import PETScSolver
from retropy.benchmarks import EllipticTransportBenchmark

from utility_functions import convergence_rate

from math import isclose


class DG0SteadyTransportTest(EllipticTransportBenchmark, DG0Kernel, PETScSolver):
    def __init__(self, nx):
        marked_mesh = self.get_mesh_and_markers(nx, "triangle")
        super().__init__(marked_mesh)

        self.set_flow_field()
        self.define_problem()
        self.generate_solver()
        self.set_solver_parameters(linear_solver="gmres", preconditioner="jacobi")


list_of_nx = [10, 20]
element_diameters = []
err_norms = []

for nx in list_of_nx:
    problem = DG0SteadyTransportTest(nx)
    problem.solve_transport()
    problem.get_solution()
    error_norm = problem.get_error_norm()
    err_norms.append(error_norm)
    element_diameters.append(problem.get_mesh_characterisitic_length())

convergence_rate_m = convergence_rate(err_norms, element_diameters)

print(convergence_rate_m)


def test_function():
    assert isclose(convergence_rate_m[0], 1, rel_tol=0.2)
