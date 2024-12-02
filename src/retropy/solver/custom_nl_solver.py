# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import *
from petsc4py import PETSc
import numpy as np

class CustomNLSolver(PETScSolver):
    """A solver class that is used as a mixin for problem classes."""

    num_forms = 1

    def generate_solver(self, eval_jacobian=True):
        """"""

        self.__func_space = self.get_function_space()

        self.__u0 = self.get_fluid_components()
        self.__u1 = Function(self.__func_space)
        self.__du = TrialFunction(self.__func_space)

        self.add_time_derivatives(self.__u0)
        self.add_physics_to_form(self.__u0)

        self.__forms = self.get_forms()
        self.__form = self.__forms[0]

        self.__form = action(self.__form, self.__u1)

        if eval_jacobian==True:
            self.jacobian = derivative(self.__form, self.__u1, self.__du)

        self._TransientSolver__u1 = self.__u1

        b = PETScVector()
        J_mat = PETScMatrix()

        self._dummy_u = Function(self.__func_space)

        self.snes = PETSc.SNES().create(MPI.comm_world)
        self.snes.setFunction(self.__F, b.vec())
        self.snes.setJacobian(self.__J, J_mat.mat())

    def __F(self, snes, x, F):
        F = PETScVector(F)
        u = self.__u1
        u.vector()[:] = np.exp(u.vector()[:])
        x.copy(u.vector().vec())
        u.vector().apply("")

        assemble(self.__form, tensor=F)
        for bc in self.get_dirichlet_bcs():
            bc.apply(F, x)
            bc.apply(F, u.vector())

    def __J(self, snes, x, J, P):
        J = PETScMatrix(J)
        u = self.__u1
        x.copy(u.vector().vec())
        u.vector().apply("")

        assemble(self.jacobian, tensor=J)

        for bc in self.get_dirichlet_bcs():
            bc.apply(J)

    def solve_one_step(self):
        self.snes.solve(None, self.__u1.vector().vec())

        return self.snes.getConvergedReason() > 0

    def evaluate_jacobian(self, form):
        self.jacobian = derivative(action(form, self.__u1), self.__u1, self.__du)

    def get_solver_parameters(self):
        return PETSc.Options()

    def set_solver_parameters(self, linear_solver='gmres', preconditioner='jacobi'):
        opts = self.get_solver_parameters()

        opts['snes_monitor'] = None
        opts['snes_linesearch_monitor'] = None
        opts['snes_converged_reason'] = None
        opts['snes_type'] = 'newtonls'
        opts['snes_linesearch'] = 'bt'

        opts['ksp_monitor_true_residual'] = None
        opts['ksp_max_it'] = 500
        opts['ksp_rtol'] = 1e-12
        opts['ksp_atol'] = 1e-14
        opts['ksp_converged_reason'] = None

        opts['pc_hypre_boomeramg_strong_threshold'] = 0.4
        opts['pc_hypre_boomeramg_truncfactor'] = 0.0
        opts['pc_hypre_boomeramg_print_statistics'] = 0

        self.snes.setFromOptions()

        self.snes.setTolerances(rtol=1e-10, atol=1e-12, max_it=50)
        self.snes.getKSP().setType('bcgs')
        self.snes.getKSP().getPC().setType('hypre')
        self.snes.getKSP().getPC().setHYPREType('boomeramg')
