# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import *

class TransientSolver:
    """A solver class that is used as a mixin for problem classes."""

    num_forms = 1

    def generate_solver(self):
        """"""

        self.__func_space = self.get_function_space()

        self.__u0 = self.get_fluid_components()
        self.__u1 = Function(self.__func_space)

        self.add_physics_to_form(self.__u0, kappa=Constant(self.mesh, 1.0), f_id=0)
        self.add_time_derivatives(self.__u0)
        self.__forms = self.get_forms()
        self.__form = self.__forms[0]

        a, L = lhs(self.__form), rhs(self.__form)

        self.__problem = LinearProblem(a, L, self.get_dirichlet_bcs(), self.__u1)

    def set_solver_parameters(self, linear_solver='gmres', preconditioner='jacobi'):
        prm = self.__problem.solver
        prm.setType(linear_solver)
        prm.getPC().setType(preconditioner)

        set_default_solver_parameters(prm)

        return prm

    def set_dt(self, dt_val):
        self.dt.value = dt_val

    def solve_one_step(self):
        self.__problem.solve()

    def get_solution(self):
        return self.__u1

    def assign_u1_to_u0(self):
        self.fluid_components.x.array[:] = self.__u1.x.array

    def assign_u0_to_u1(self):
        self.__u1.x.array[:] = self.fluid_components.x.array

    def solve_transport(self, dt_val=1.0, timesteps=1):
        """"""

        self.set_dt(dt_val)

        current_time = 0.0
        self.save_to_file(time=current_time)

        for i in range(timesteps):
            self.solve_one_step()
            self.assign_u1_to_u0()

            current_time += dt_val
            self.save_to_file(time=current_time)
