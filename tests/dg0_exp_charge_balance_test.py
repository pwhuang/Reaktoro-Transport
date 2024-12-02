# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os

os.environ["OMP_NUM_THREADS"] = "1"

from retropy.problem import TracerTransportProblemExp
from retropy.physics import DG0Kernel
from retropy.solver import TransientNLSolver
from retropy.benchmarks import ChargeBalancedDiffusion

from dolfinx.fem import Constant
import matplotlib.pyplot as plt


def set_default_solver_parameters(prm):
    prm.convergence_criterion = "residual"
    prm.atol = 1e-12
    prm.rtol = 1e-14
    prm.max_it = 1000
    prm.nonzero_initial_guess = True
    prm.report = True


class DG0ExpChargeBalanceTest(
    TracerTransportProblemExp, ChargeBalancedDiffusion, DG0Kernel, TransientNLSolver
):
    def __init__(self, nx, t0):
        marked_mesh = self.get_mesh_and_markers(nx)
        super().__init__(marked_mesh)

        self.set_flow_field()
        self.define_problem(t0=t0)

        self.generate_solver()
        self.set_solver_parameters(linear_solver="gmres", preconditioner="jacobi")

    def set_solver_parameters(self, linear_solver="gmres", preconditioner="jacobi"):
        super().set_solver_parameters(linear_solver, preconditioner)

        set_default_solver_parameters(self.get_solver())

    def add_physics_to_form(self, u, kappa, f_id):
        super().add_physics_to_form(u)

        theta = Constant(self.mesh, 0.5)
        one = Constant(self.mesh, 1.0)

        self.add_explicit_charge_balanced_diffusion(u, kappa=one - theta, marker=0)
        self.add_implicit_charge_balanced_diffusion(kappa=theta, marker=0)

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
    problem = DG0ExpChargeBalanceTest(nx, t0)
    problem.solve_transport(dt_val=dt, timesteps=timesteps[i])

    t_end = timesteps[i] * dt + t0
    problem.get_solution(t_end)
    error_norm = problem.get_error_norm()
    err_norms.append(error_norm)

    # problem.mpl_output()

print(err_norms)


def test_function():
    assert err_norms[-1] < 1e-2
