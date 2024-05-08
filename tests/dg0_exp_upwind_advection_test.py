# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os

os.environ["OMP_NUM_THREADS"] = "1"

from retropy.problem import TracerTransportProblemExp
from retropy.physics import DG0Kernel
from retropy.solver import TransientNLSolver
from retropy.manager import XDMFManager
from retropy.benchmarks import RotatingCone

import numpy as np


class DG0ExpUpwindAdvectionTest(
    TracerTransportProblemExp, RotatingCone, DG0Kernel, TransientNLSolver, XDMFManager
):
    def __init__(self, nx, is_output):
        marked_mesh = self.get_mesh_and_markers(nx, "quadrilateral")
        super().__init__(marked_mesh)

        self.set_flow_field()
        self.define_problem()
        self.generate_solver()
        self.set_solver_parameters(linear_solver="gmres", preconditioner="jacobi")

        if is_output == True:
            self.generate_output_instance("rotating_cone_exp")

    def guess_solution(self):
        self.get_solver_u1().x.array[:] = 0.1 * self.fluid_components.x.array

nx = 50
dt = 1e-2
timesteps = 100

problem = DG0ExpUpwindAdvectionTest(nx, is_output=False)

initial_mass = problem.get_total_mass()
initial_center_x, initial_center_y = problem.get_center_of_mass()
problem.solve_transport(dt_val=dt, timesteps=timesteps)

problem.get_solution()
advected_mass = problem.get_total_mass()
advected_center_x, advected_center_y = problem.get_center_of_mass()

mass_error = np.abs(initial_mass - advected_mass)
center_of_mass_error = (
    (advected_center_x - initial_center_x) ** 2
    + (advected_center_y - initial_center_y) ** 2
) ** 0.5

allowed_mass_error = 1e-10
allowed_drift_distance = 0.05

print(mass_error, center_of_mass_error)


def test_function():
    assert (
        mass_error < allowed_mass_error
        and center_of_mass_error < allowed_drift_distance
    )
