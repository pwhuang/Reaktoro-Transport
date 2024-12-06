# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.problem import DOLFIN_EPS
from retropy.benchmarks import EllipticTransportBenchmark
from dolfinx.fem import Function, FunctionSpace, Constant, assemble_scalar, form
from dolfinx.fem.petsc import assemble_vector

from mpi4py import MPI
import numpy as np
import ufl
from ufl import dot, avg, jump
from ufl.operators import elem_div


class RotatingCone(EllipticTransportBenchmark):
    """This benchmark problem is inspired by Randall J. Leveque's work:
    High-resolution conservative algorithms for advection in incompressible
    flow, published in SIAM Journal on Numerical Analysis.
    doi: doi.org/10.1137/0733033
    """

    @staticmethod
    def solution_expr(x):
        x0, x1, sigma = 0.5, 0.8, 0.2
        return np.where(
            ((x[0] - x0) / sigma) ** 2 + ((x[1] - x1) / sigma) ** 2 < 0.1,
            np.cos(np.pi * (x[0] - x0) / sigma) * np.cos(np.pi * (x[1] - x1) / sigma),
            DOLFIN_EPS,
        )

    def set_flow_field(self):
        expr = lambda x: (
            np.sin(np.pi * x[0]) * np.sin(np.pi * x[0]) * np.sin(2 * np.pi * x[1]),
            -np.sin(np.pi * x[1]) * np.sin(np.pi * x[1]) * np.sin(2 * np.pi * x[0]),
        )

        self.velocity_func_space = FunctionSpace(self.mesh, ('RT', 1))

        self.fluid_velocity = Function(self.velocity_func_space)
        self.fluid_velocity.interpolate(expr)

        self.fluid_pressure = Function(self.DG0_space)

    def set_advection_velocity(self):
        zero = Constant(self.mesh, 0.0)
        self.advection_velocity = ufl.as_vector(
            [
                ufl.cos(ufl.pi * self.current_time) * self.fluid_velocity if is_mobile else zero * self.fluid_velocity
                for is_mobile in self.component_mobility
            ]
        )

    def define_problem(self):
        self.set_components("solute")
        self.set_component_fe_space()
        self.set_component_mobility([True])
        self.set_advection_velocity()

        self.initialize_form()
        self.mark_component_boundary({"solute": self.marker_dict.values()})

        self.set_component_ics("solute", self.solution_expr)

    def add_physics_to_form(self, u, kappa, f_id):
        self.add_time_derivatives(u)
        self.add_explicit_advection(u, kappa=kappa, marker=0, f_id=f_id)

    @staticmethod
    def flux_limiter(r):
        return np.maximum(0.0, np.minimum(1.5, r))
        
    def add_corrector_to_form(self, u1, f_id):
        # TODO: Implement and test flux-limited advection in 2D.
        pass
        
    def solve_limiter_function(self):
        # TODO: Implement and test flux-limited advection in 2D.
        pass

    def get_solution(self):
        self.solution = Function(self.func_space_list[0])
        self.solution.interpolate(self.solution_expr)

        return self.solution.copy()

    def get_total_mass(self):
        comm = self.mesh.comm
        self.total_mass = assemble_scalar(form(self.fluid_components.sub(0) * self.dx))
        self.total_mass = comm.allreduce(self.total_mass, op=MPI.SUM)

        return self.total_mass

    def get_center_of_mass(self):
        comm = self.mesh.comm
        center_x = assemble_scalar(
            form(self.fluid_comp_sub[0] * self.cell_coord.sub(0) * self.dx)
        )
        center_y = assemble_scalar(
            form(self.fluid_comp_sub[0] * self.cell_coord.sub(1) * self.dx)
        )

        center_x = comm.allreduce(center_x, op=MPI.SUM)
        center_y = comm.allreduce(center_y, op=MPI.SUM)

        return center_x / self.total_mass, center_y / self.total_mass
