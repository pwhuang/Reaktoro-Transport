# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os

os.environ["OMP_NUM_THREADS"] = "1"

from retropy.physics import DG0Kernel
from retropy.solver import PETScSolver
from retropy.benchmarks import TracerBreakthrough

from utility_functions import convergence_rate

from math import isclose
import matplotlib.pyplot as plt


class DG0BreakthroughTest(TracerBreakthrough, DG0Kernel, PETScSolver):
    def __init__(self, nx, Pe):
        super().__init__(self.get_mesh_and_markers(nx))

        self.define_problem(Peclet_number=Pe)
        self.set_flow_field()
        self.generate_solver()
        self.set_solver_parameters(linear_solver="gmres", preconditioner="jacobi")

    def solve_one_step(self):
        return self._problem.solve_without_matrix_assembly()

    def mpl_output(self):
        x_space = self.cell_coord.x.array
        numerical_solution = self.fluid_components.x.array
        analytical_solution = self.solution.x.array

        _, ax = plt.subplots(1, 1)
        ax.plot(x_space, analytical_solution, lw=3, c="C0")
        ax.plot(x_space, numerical_solution, ls=(0, (5, 5)), lw=2, c="C1")
        plt.show()


Pe = 113.0

nx_list = [33, 66]
dt_list = [2.6e-2, 1.3e-2]
timesteps = [15, 30]
err_norms = []

for nx, dt, timestep in zip(nx_list, dt_list, timesteps):
    problem = DG0BreakthroughTest(nx, Pe)
    problem.solve_transport(dt_val=dt, timesteps=timestep)

    t_end = timestep * dt
    problem.get_solution(t_end)
    error_norm = problem.get_error_norm()
    err_norms.append(error_norm)

print(err_norms)

convergence_rate_m = convergence_rate(err_norms, dt_list)
print(convergence_rate_m)


def test_function():
    assert isclose(convergence_rate_m[0], 1, rel_tol=0.2)
