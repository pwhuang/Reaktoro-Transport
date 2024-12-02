# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os

os.environ["OMP_NUM_THREADS"] = "1"

from retropy.physics import DG0Kernel
from retropy.solver import PETScSolver
from retropy.manager import XDMFManager
from retropy.benchmarks import ReactingSpecies


class DG0ReactiveTransportTest(
    ReactingSpecies, DG0Kernel, PETScSolver, XDMFManager
):
    def __init__(self, nx, is_output):
        marked_mesh = self.get_mesh_and_markers(nx, "triangle")
        super().__init__(marked_mesh)

        self.set_flow_field()
        self.define_problem()
        self.generate_solver()
        self.set_solver_parameters(linear_solver="gmres", preconditioner="jacobi")

        if is_output == True:
            self.generate_output_instance("reacting_species")


nx = 20
list_of_dt = [1e-1]
timesteps = [10]
err_norms = []

for i, dt in enumerate(list_of_dt):
    problem = DG0ReactiveTransportTest(nx, is_output=False)
    problem.solve_transport(dt_val=dt, timesteps=timesteps[i])
    problem.generate_solution()

    error_norm = problem.get_error_norm()
    err_norms.append(error_norm)

# TODO: Figure out how to setup and benchmark this problem properly.
# By calling the mpl_output function, one can see the numerical solution
# resembles the analytical the solution. However, the error says otherwise.
# This is a non-urgent issue. Please address this when you are interested in
# this particular problem.
# problem.mpl_output()
print(err_norms)


def test_function():
    assert err_norms[-1] < 1.0
