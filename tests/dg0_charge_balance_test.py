# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os

os.environ["OMP_NUM_THREADS"] = "1"

from retropy.physics import DG0Kernel
from retropy.solver import PETScSolver
from retropy.manager import XDMFManager
from retropy.benchmarks import ChargeBalancedDiffusion

import matplotlib.pyplot as plt
from dolfinx.fem import Constant


class DG0ChargeBalanceTest(
    ChargeBalancedDiffusion, DG0Kernel, PETScSolver, XDMFManager
):
    def __init__(self, nx, t0, is_output=False):
        super().__init__(self.get_mesh_and_markers(nx))

        self.set_flow_field()
        self.define_problem(t0=t0)
        self.generate_solver()
        self.set_solver_parameters(linear_solver="gmres", preconditioner="jacobi")

        if is_output == True:
            self.generate_output_instance("charge_balance")

    def add_physics_to_form(self, u, **kwargs):
        super().add_physics_to_form(u, **kwargs)

        theta = Constant(self.mesh, 0.5)
        one = Constant(self.mesh, 1.0)

        self.add_explicit_charge_balanced_diffusion(u, kappa=one - theta, marker=0)
        self.add_semi_implicit_charge_balanced_diffusion(u, kappa=theta, marker=0)

    def mpl_output(self):
        x_space = self.cell_coord.x.array
        numerical_solution = self.fluid_components.x.array.reshape(-1, 2)
        analytical_solution = self.solution.x.array.reshape(-1, 2)[:, 0]

        _, ax = plt.subplots(1, 1)
        ax.plot(x_space, analytical_solution, lw=3, c="C0")
        ax.plot(x_space, numerical_solution[:, 0], ls=(0, (5, 5)), lw=2, c="C1")
        ax.plot(x_space, numerical_solution[:, 1], ls=(2.5, (5, 5)), lw=2, c="C3")
        plt.show()


nx, t0 = 51, 1.0
list_of_dt = [3e-1]
timesteps = [10]
err_norms = []

for i, dt in enumerate(list_of_dt):
    problem = DG0ChargeBalanceTest(nx, t0, is_output=False)
    problem.solve_transport(dt_val=dt, timesteps=timesteps[i])

    t_end = timesteps[i] * dt + t0
    problem.get_solution(t_end)
    error_norm = problem.get_error_norm()
    err_norms.append(error_norm)

print(err_norms)


def test_function():
    assert err_norms[-1] < 1e-2
