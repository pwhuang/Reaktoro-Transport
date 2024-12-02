# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.mesh import MarkedLineMesh
from retropy.problem import TracerTransportProblem, DOLFIN_EPS

from dolfinx.fem import FunctionSpace, Function, Constant, assemble_scalar, form
from ufl.algebra import Abs

import numpy as np
from scipy.special import erfc
from mpi4py import MPI


class TracerBreakthrough(TracerTransportProblem):
    """
    This benchmark problem tests a tracer breakthrough problem based on van Genuchten, 1980 and Brenner, 1962.
    We consider a finite domain with a third-type condition at the inlet and an outflow condition at the outlet.

    van Genuchten, 1980: Determining transport parameters from solute displacement experiments,
    Research Report No. 118, U.S. Salinity Laboaratory
    Brenner, 1962: The diffusion model of longitudinal mixing in beds of finite length: numerical values,
    Chemical Engineering Science, 17, 229--243
    """

    @staticmethod
    def solution_expr(t, L, R, v, D):
        return (
            lambda x: 0.5 * erfc(0.5 * (R * x[0] - v * t) / (D * R * t) ** 0.5)
            + ((v**2 * t) / (np.pi * D * R)) ** 0.5
            * np.exp(-((R * x[0] - v * t) ** 2) / (4.0 * D * R * t))
            - 0.5
            * (1 + v * x[0] / D + v**2 * t / (D * R))
            * np.exp(v * x[0] / D)
            * erfc((R * x[0] + v * t) / (4.0 * D * R * t) ** 0.5)
            + ((4 * v**2 * t) / (np.pi * D * R)) ** 0.5
            * (1 + 0.25 * v * (2 * L - x[0] + v * t / R) / D)
            * np.exp(v * L / D - R * (2 * L - x[0] + v * t / R) ** 2 / (4 * D * t))
            - v
            / D
            * (
                2 * L
                - x[0]
                + 1.5 * v * t / R
                + 0.25 * v * (2 * L - x[0] + v * t / R) ** 2 / D
            )
            * np.exp(v * L / D)
            * erfc((R * (2 * L - x[0]) + v * t) / (4.0 * D * R * t) ** 0.5)
        )

    @staticmethod
    def initial_expr():
        return lambda x: 0.0 * x[0]

    def get_mesh_and_markers(self, nx):
        marked_mesh = MarkedLineMesh(xmin=0.0, xmax=1.0, num_elements=nx)
        self.mesh_characteristic_length = 1.0 / nx

        return marked_mesh

    def get_mesh_characterisitic_length(self):
        return self.mesh_characteristic_length

    def set_flow_field(self):
        self.fluid_velocity = Function(self.Vec_CG1_space)
        self.fluid_velocity.x.array[:] = 1.0
        self.fluid_velocity.x.scatter_forward()
        self.set_advection_velocity()

    def define_problem(self, Peclet_number):
        self.set_components("C")
        self.set_component_fe_space()
        self.initialize_form()

        self.Pe = Peclet_number
        self.set_molecular_diffusivity([1.0 / Peclet_number])

        self.mark_component_boundary(
            {"C": [self.marker_dict["left"]], 
             "outlet": [self.marker_dict["right"]]}
        )

        self.set_component_ics("C", self.initial_expr())

    def add_physics_to_form(self, u, **kwargs):
        one = Constant(self.mesh, 1.0)
        self.inlet_flux = Constant(self.mesh, -1.0)

        self.add_time_derivatives(u)
        self.add_explicit_advection(u, kappa=one, marker=0)
        self.add_implicit_diffusion("C", kappa=one, marker=0)

        self.add_component_flux_bc("C", [self.inlet_flux], kappa=one)
        self.add_outflow_bc(u)

    def get_solution(self, t_end):
        self.interpolation_space = FunctionSpace(self.mesh, ('CG', 2)) 
        expr = self.solution_expr(t_end, L=1.0, R=1.0, v=1.0, D=1.0 / self.Pe)

        self.u_solution = Function(self.interpolation_space)
        self.u_solution.interpolate(expr)

        self.solution = Function(self.comp_func_spaces)
        self.solution.sub(0).interpolate(self.u_solution)

        return self.solution

    def get_error_norm(self):
        u_numerical = Function(self.interpolation_space)
        u_numerical.interpolate(self.fluid_components.sub(0))

        comm = self.mesh.comm
        mass_error = u_numerical - self.u_solution
        mass_error_norm = assemble_scalar(form(mass_error**2 * self.dx))

        return np.sqrt(comm.allreduce(mass_error_norm, op=MPI.SUM))
