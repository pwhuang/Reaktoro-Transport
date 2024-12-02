# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from mpi4py import MPI

import os
os.environ['OMP_NUM_THREADS'] = '1'

from retropy.physics import DG0Kernel
from retropy.solver import PETScSolver
from retropy.mesh import XDMFMesh, MarkedRectangleMesh
from retropy.benchmarks import DiffusionBenchmark

from utility_functions import convergence_rate

from dolfinx.fem import Constant
import numpy as np
from math import isclose

class XDMFRectangleMesh(MarkedRectangleMesh, XDMFMesh):
    def __init__(self):
        super().__init__()

class DiffusionBenchmark(DiffusionBenchmark):
    def get_mesh_and_markers(self, filename: str, meshname: str):
        marked_mesh = XDMFRectangleMesh()
        marked_mesh.read_mesh(filename, meshname)

        x_min = marked_mesh.mesh.geometry.x[:, 0].min()[np.newaxis]
        x_max = marked_mesh.mesh.geometry.x[:, 0].max()[np.newaxis]
        y_min = marked_mesh.mesh.geometry.x[:, 1].min()[np.newaxis]
        y_max = marked_mesh.mesh.geometry.x[:, 1].max()[np.newaxis]

        comm = marked_mesh.mesh.comm
        comm.Allreduce(MPI.IN_PLACE, x_min, op=MPI.MIN)
        comm.Allreduce(MPI.IN_PLACE, x_max, op=MPI.MAX)
        comm.Allreduce(MPI.IN_PLACE, y_min, op=MPI.MIN)
        comm.Allreduce(MPI.IN_PLACE, y_max, op=MPI.MAX)

        marked_mesh.set_bottom_left_coordinates(coord_x = x_min[0], coord_y = y_min[0])
        marked_mesh.set_top_right_coordinates(coord_x = x_max[0], coord_y = y_max[0])
        marked_mesh.locate_and_mark_boundaries()

        marked_mesh.generate_boundary_markers()
        marked_mesh.generate_interior_markers()
        marked_mesh.generate_domain_markers()

        return marked_mesh

class DG0SteadyDiffusionReadMeshTest(DiffusionBenchmark, DG0Kernel, PETScSolver):
    def __init__(self, filename):
        marked_mesh = self.get_mesh_and_markers(filename, meshname='Grid')
        super().__init__(marked_mesh)

        self.set_flow_field()
        self.define_problem()
        self.set_problem_bc()

        self.generate_solver()
        self.set_solver_parameters(linear_solver='gmres', preconditioner='jacobi')

    def set_problem_bc(self):
        values = DiffusionBenchmark.set_problem_bc(self)
        # When solving steady-state problems, the diffusivity of the diffusion
        # boundary is a penalty term to the variational form.
        self.add_component_diffusion_bc('solute', diffusivity=Constant(self.mesh, 1e3),
                                        kappa=Constant(self.mesh, 1.0), values=values)

list_of_filenames = ['mesh/2d_1e-1.xdmf', 'mesh/2d_5e-2.xdmf']
element_diameters = [1e-1, 5e-2]
err_norms = []

for filename in list_of_filenames:
    problem = DG0SteadyDiffusionReadMeshTest(filename)
    problem.solve_transport()
    problem.get_solution()
    error_norm = problem.get_error_norm()
    err_norms.append(error_norm)

print(err_norms)

convergence_rate_m = convergence_rate(err_norms, element_diameters)
print(convergence_rate_m)

def test_function():
    assert isclose(convergence_rate_m[0], 1, rel_tol=0.2)
