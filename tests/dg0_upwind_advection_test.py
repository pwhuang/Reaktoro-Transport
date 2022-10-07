# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os
os.environ['OMP_NUM_THREADS'] = '1'

from retropy.physics import DG0Kernel
from retropy.solver import TransientSolver
from retropy.manager import XDMFManager

from benchmarks import RotatingCone

from math import isclose

class DG0UpwindAdvectionTest(RotatingCone, DG0Kernel, TransientSolver,
                             XDMFManager):
    def __init__(self, nx, is_output=False):
        super().__init__(*self.get_mesh_and_markers(nx, 'triangle'))

        self.set_flow_field()
        self.define_problem()
        self.generate_solver()
        self.set_solver_parameters(linear_solver='gmres', preconditioner='amg')

        if is_output==True:
            self.generate_output_instance('rotating_cone')

    def solve_transport(self, dt_val, timesteps):
        self.dt.assign(dt_val)
        endtime = 0.0

        self.save_to_file(time=endtime)

        for i in range(timesteps):
            self.solve_one_step()
            self.assign_u1_to_u0()
            endtime += dt_val
            self.t_end.assign(endtime)
            self.save_to_file(time=endtime)

        self.delete_output_instance()

nx = 30
list_of_dt = [1e-2]
timesteps = [100]
err_norms = []

for i, dt in enumerate(list_of_dt):
    problem = DG0UpwindAdvectionTest(nx, is_output=False)
    initial_mass = problem.get_total_mass()
    initial_center_x, initial_center_y = problem.get_center_of_mass()
    problem.solve_transport(dt_val=dt, timesteps=timesteps[i])

    problem.get_solution()
    error_norm = problem.get_error_norm()
    err_norms.append(error_norm)
    advected_mass = problem.get_total_mass()
    advected_center_x, advected_center_y = problem.get_center_of_mass()

mass_error = abs(initial_mass-advected_mass)
center_of_mass_error = ((advected_center_x - initial_center_x)**2 - \
                        (advected_center_y - initial_center_y)**2)**0.5

allowed_mass_error = 1e-10
allowed_drift_distance = 0.05

print(mass_error, center_of_mass_error)

def test_function():
    assert mass_error < allowed_mass_error and \
           center_of_mass_error < allowed_drift_distance
