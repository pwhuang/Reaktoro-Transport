# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os

os.environ["OMP_NUM_THREADS"] = "1"

from retropy.physics import DG0Kernel
from retropy.solver import PETScSolver
from retropy.benchmarks import ParticleAttachment

from dolfinx.fem import Function
from utility_functions import convergence_rate

from math import isclose
import numpy as np
import matplotlib.pyplot as plt


class DG0ParticleReversibleAttachmentTest(ParticleAttachment, DG0Kernel, PETScSolver):
    def __init__(self, nx, Pe, Da_att, Da_det, M, t0):
        super().__init__(self.get_mesh_and_markers(nx))

        self.define_problem(Pe, Da_att, Da_det, M, t0)
        self.set_flow_field()
        self.generate_solver()
        self.set_solver_parameters(linear_solver="gmres", preconditioner="jacobi")

        self.C_of_t = []
        self.t_space = []

    def save_to_file(self, time):
        self.t_space.append(time.copy())
        self.C_of_t.append(self.fluid_components.x.array.reshape(-1, 2).T[0].copy())

    def generate_solution(self):
        x_space = self.cell_coord.x.array
        t, t0 = self.current_time.value, self.t0.value
        Da_att, Da_det, M = self.Da_att.value, self.Da_det.value, self._M.value

        self.solution = Function(self.comp_func_spaces)
        solution = np.zeros_like(self.solution.x.array)

        C, S = self.reversible_attachment(x_space, [t], t0, M, Da_att, Da_det)

        solution[::2] = C[:, 0]
        solution[1::2] = S[:, 0]

        self.solution.x.array[:] = solution
        self.solution.x.scatter_forward()

        self.C_btc, _ = self.reversible_attachment(
            [1.0], self.t_space, t0, M, Da_att, Da_det
        )
        self.C_btc = self.C_btc[0]

    def get_btc_error_norm(self):
        return np.sqrt(
            np.sum((np.array(self.C_of_t)[:, -1] - self.C_btc) ** 2 * self.dt.value)
        )

    def mpl_output(self):
        x_space = self.cell_coord.x.array
        numerical_solution = self.fluid_components.x.array.reshape(-1, 2).T
        analytical_solution = self.solution.x.array.reshape(-1, 2).T
        

        _, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(
            x_space,
            analytical_solution[0],
            lw=3,
            c="C0",
            alpha=0.7,
            label="analytical, C",
        )
        ax[0].plot(
            x_space,
            analytical_solution[1],
            lw=3,
            c="C0",
            alpha=0.7,
            label="analytical, S",
        )
        ax[0].plot(
            x_space,
            numerical_solution[0],
            ls=(0, (3, 3)),
            lw=2,
            c="C1",
            label="numerical, C",
        )
        ax[0].plot(
            x_space,
            numerical_solution[1],
            ls=(0, (3, 3)),
            lw=2,
            c="C2",
            label="numerical, S",
        )
        ax[0].set_xlabel("Location (-)")
        ax[0].legend()

        ax[1].plot(self.t_space, self.C_btc, label="analytical")
        ax[1].plot(self.t_space, np.array(self.C_of_t)[:, -1], label="numerical")
        ax[1].set_xlabel("PV (-)")
        ax[1].legend()
        ax[1].set_title("Breakthrough curve")
        plt.tight_layout()
        plt.show()


Pe, Da_att, Da_det, M, t0 = np.inf, 5.5, 1.3, 2.2, 1.0

nx_list = [33, 66]
dt_list = [2.0e-2, 1.0e-2]
timesteps = [50, 100]
err_norms = []

for nx, dt, timestep in zip(nx_list, dt_list, timesteps):
    problem = DG0ParticleReversibleAttachmentTest(nx, Pe, Da_att, Da_det, M, t0)
    problem.solve_transport(dt_val=dt, timesteps=timestep)
    problem.inlet_flux.value = 0.0
    problem.solve_transport(dt_val=dt, timesteps=int(6.0 * timestep))

    problem.generate_solution()
    error_norm = problem.get_btc_error_norm()
    err_norms.append(error_norm)

    # problem.mpl_output()

print(err_norms)

convergence_rate_m = convergence_rate(err_norms, dt_list)
print(convergence_rate_m)


def test_function():
    assert isclose(convergence_rate_m[0], 1, rel_tol=0.2)
