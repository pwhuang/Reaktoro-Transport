# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os
os.environ['OMP_NUM_THREADS'] = '1'

from retropy.mesh import MarkedRectangleMesh
from retropy.problem import TracerTransportProblem, DarcyFlowUzawa
from retropy.physics import DG0Kernel
from retropy.solver import PETScSolver
from retropy.manager import XDMFManager as OutputManager

from dolfinx.fem import Constant, Function
from petsc4py.PETSc import ScalarType
from ufl import as_vector
import numpy as np

DOLFIN_EPS = 1e-16

class MeshFactory(MarkedRectangleMesh):
    def __init__(self, nx, ny, mesh_type, mesh_shape):
        self.set_bottom_left_coordinates(coord_x = 0.0, coord_y = 1e-15)
        self.set_top_right_coordinates(coord_x = 4.0, coord_y = 1.0 + 1e-15)
        self.set_number_of_elements(nx, ny)
        self.set_mesh_type(mesh_type)
        self.locate_and_mark_boundaries()

        self.generate_mesh(mesh_shape)
        self.generate_boundary_markers()
        self.generate_interior_markers()
        self.generate_domain_markers()

    def locate_and_mark_boundaries(self, boundary_eps=1e-8):
        right_marker = lambda x: np.isclose(x[0], self.xmax, atol=boundary_eps)
        top_marker = lambda x: np.isclose(x[1], self.ymax, atol=boundary_eps)
        left_marker = lambda x: np.isclose(x[0], self.xmin, atol=boundary_eps)

        bottom_inner_marker = lambda x: np.logical_and.reduce(
            (np.isclose(x[1], self.ymin, atol=boundary_eps),
            x[0] < 3.0 + boundary_eps,
            x[0] > 1.0 - boundary_eps))

        bottom_outer_marker = lambda x: np.logical_and(
            np.isclose(x[1], self.ymin, atol=boundary_eps),
            np.logical_or(x[0] > 3.0 - boundary_eps,
                          x[0] < 1.0 + boundary_eps))

        self.marker_dict = {"right": 1, "top": 2, "left": 3, "bottom_inner": 4, "bottom_outer": 5}

        self.locator_dict = {
            "right": right_marker,
            "top": top_marker,
            "left": left_marker,
            "bottom_inner": bottom_inner_marker,
            "bottom_outer": bottom_outer_marker
        }

        return self.marker_dict, self.locator_dict

class FlowManager(DarcyFlowUzawa):
    def set_fluid_properties(self):
        self.set_porosity(1.0)
        self.set_permeability(1.0)
        self.set_fluid_density(1.0)
        self.set_fluid_viscosity(1.0)
        self.set_gravity((0.0, 0.0))

    def setup_flow_solver(self):
        self.set_pressure_fe_space('DG', 0)
        self.set_velocity_fe_space('RTCF', 1)

        self.set_fluid_properties()

        self.generate_form()
        self.generate_residual_form()
        self.fluid_pressure.interpolate(lambda x: 0.0 * x[0])
        self.set_pressure_bc({})

        zero = Constant(self.mesh, ScalarType(0.0))
        self.add_momentum_source([as_vector([zero, self.fluid_components[0]])])

        velocity_bc = Function(self.velocity_func_space)
        velocity_bc.x.array[:] = 0.0
        velocity_bc.x.scatter_forward()
        self.set_velocity_bc({'top': velocity_bc,
                              'right': velocity_bc,
                              'bottom_inner': velocity_bc,
                              'bottom_outer': velocity_bc,
                              'left': velocity_bc})

        self.set_additional_parameters(r_val=4.0, omega_by_r=1.0)
        self.assemble_matrix()
        self.set_flow_solver_params()

class TransportManager(TracerTransportProblem, DG0Kernel, PETScSolver):
    def __init__(self, nx, ny, mesh_type, mesh_shape):
        marked_mesh = MeshFactory(nx, ny, mesh_type, mesh_shape)
        TracerTransportProblem.__init__(self, marked_mesh)

    def define_problem(self, Rayleigh_number=400.0):
        self.set_components('Temp')
        self.set_component_fe_space()
        self.initialize_form(num_forms=1)

        self.set_molecular_diffusivity([1.0])
        self.mark_component_boundary({'Temp': [self.marker_dict['top'],
                                               self.marker_dict['bottom_inner']]})

        self.temp_bc = [Constant(self.mesh, ScalarType(0.0)),
                        Constant(self.mesh, ScalarType(1.0))]

        self.set_component_ics('Temp', lambda x: 0.0 * x[0])
        self.Ra = Constant(self.mesh, Rayleigh_number)

    def add_physics_to_form(self, u, kappa, f_id=0):
        self.add_time_derivatives(u, kappa, f_id=f_id)
        self.add_explicit_advection(u, kappa=kappa, marker=0, f_id=f_id)

        for component in self.component_dict.keys():
            self.add_implicit_diffusion(component, kappa=kappa, marker=0, f_id=f_id)
            self.add_component_diffusion_bc(component, Constant(self.mesh, ScalarType(1e2)),\
                                            self.temp_bc, kappa, f_id)

    def setup_transport_solver(self):
        self.fluid_velocity = self.Ra * self.fluid_velocity
        self.set_advection_velocity()
        self.generate_solver()
        self.set_solver_parameters('gmres', 'jacobi')

    def solve_transport(self):
        self.solve_one_step()
        self.assign_u1_to_u0()

class ElderProblem(TransportManager, FlowManager, OutputManager):
    """
    This is an example of solving the Elder problem using the Boussinesq approx-
    imation. Note that we set the fluid density to 1 and utilize the
    add_momentum_source method to apply the body force term (a function of tempera-
    ture).
    """

    def __init__(self, nx, ny, mesh_type, mesh_shape):
        TransportManager.__init__(self, nx, ny, mesh_type, mesh_shape)
        self.define_problem()
        self.setup_flow_solver()
        self.setup_transport_solver()
        self.generate_output_instance('elder_problem')

    def solve(self, dt_val=1.0, timesteps=1):
        self.dt.value = dt_val
        saved_times = []

        for i in range(timesteps):
            self.solve_transport()
            self.solve_flow(target_residual=5e-9, max_steps=50)

            self.save_to_file(time=(i+1)*self.dt.value, is_saving_pv=False)
            saved_times.append((i+1)*self.dt.value)

        if self.mesh.comm.Get_rank()==0:
            np.save(self.output_file_name + '_time', np.array(saved_times), allow_pickle=False)


problem = ElderProblem(nx=80, ny=20, mesh_type="quadrilateral", mesh_shape="crossed")
problem.solve(dt_val=2e-4, timesteps=50)
