# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.mesh import MarkedLineMesh
from retropy.problem import TracerTransportProblem, DOLFIN_EPS

from dolfinx.fem import Function, Constant, assemble_scalar, form

import numpy as np
from mpi4py import MPI


class ChargeBalancedDiffusion(TracerTransportProblem):
    """
    This benchmark problem tests whether two species diffusing with different
    molecular diffusivities diffuse at the same rate when charge balance of the
    species are involved. Particularly, it demonstrates using the Crank-Nicolson
     timestepping to solve diffusion problems.
    """

    center_of_mass = 0.5

    @staticmethod
    def solution_expr(x0, t, D):
        return (
            lambda x: np.exp(-((x[0] - x0) ** 2) / (4 * D * t))
            / (4 * np.pi * D * t) ** 0.5
        )

    @staticmethod
    def initial_expr(x0, t, D):
        return (
            lambda x: np.exp(-((x[0] - x0) ** 2) / (4 * D * t))
            / np.sqrt(4 * np.pi * D * t)
            + DOLFIN_EPS
        )

    def get_mesh_and_markers(self, nx):
        marked_mesh = MarkedLineMesh(xmin=0.0, xmax=1.0, num_elements=nx)
        self.mesh_characteristic_length = 1.0 / nx

        return marked_mesh

    def get_mesh_characterisitic_length(self):
        return self.mesh_characteristic_length

    def set_flow_field(self):
        self.fluid_velocity = Function(self.Vec_CG1_space)
        self.fluid_velocity.interpolate(lambda x: (0.0 * x[0]))

    def define_problem(self, t0):
        self.set_components("Na+ Cl-")
        self.set_component_fe_space()
        self.initialize_form()

        self.D_Na = 1.33e-3
        self.D_Cl = 2.03e-3
        self.Z_Na = 1
        self.Z_Cl = -1

        self.avg_D = (abs(self.Z_Na) + abs(self.Z_Cl)) / (
            abs(self.Z_Na) / self.D_Na + abs(self.Z_Cl) / self.D_Cl
        )

        self.set_molecular_diffusivity([self.D_Na, self.D_Cl])
        self.set_charge([self.Z_Na, self.Z_Cl])
        self.set_molar_mass([1.0, 1.0])

        self.mark_component_boundary(
            {"Na+": self.marker_dict.values(), "Cl-": self.marker_dict.values()}
        )

        D = self.avg_D
        x0 = self.center_of_mass

        self.set_component_ics("Na+", self.initial_expr(x0, t0, D))
        self.set_component_ics("Cl-", self.initial_expr(x0, t0, D))

    def add_physics_to_form(self, u, **kwargs):
        theta = Constant(self.mesh, 0.5)
        one = Constant(self.mesh, 1.0)

        self.add_time_derivatives(u)
        self.add_implicit_diffusion("Na+", kappa=theta, marker=0)
        self.add_explicit_diffusion("Na+", u, kappa=one - theta, marker=0)
        self.add_implicit_diffusion("Cl-", kappa=theta, marker=0)
        self.add_explicit_diffusion("Cl-", u, kappa=one - theta, marker=0)

    def get_solution(self, t_end):
        D = self.avg_D
        x0 = self.center_of_mass
        expr = self.solution_expr(x0, t_end, D)

        self.solution = Function(self.comp_func_spaces)
        self.solution.sub(0).interpolate(expr)
        self.solution.sub(1).interpolate(expr)

        return self.solution

    def get_error_norm(self):
        comm = self.mesh.comm
        mass_error = self.fluid_components - self.solution
        mass_error_norm = assemble_scalar(form(mass_error**2 * self.dx))

        return comm.allreduce(mass_error_norm, op=MPI.SUM) ** 0.5
