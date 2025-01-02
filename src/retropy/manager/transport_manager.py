# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.problem import TracerTransportProblemExp
from retropy.physics import DG0Kernel

from dolfinx.fem import Constant
import numpy as np

class TransportManager(TracerTransportProblemExp, DG0Kernel):
    """Manages the default behavior of solving species transport."""

    def __init__(self, marked_mesh):
        super().__init__(marked_mesh)
        self.is_same_diffusivity = False

    def add_physics_to_form(self, u, kappa, f_id=0):
        """
        Explicit upwind advection and Crank-Nicolson diffusion.
        """
        theta_val = 0.5

        theta = Constant(self.mesh, theta_val)
        one = Constant(self.mesh, 1.0)

        self.add_time_derivatives(u, kappa=one, f_id=f_id)
        self.add_explicit_advection(u, kappa=one, marker=0, f_id=f_id)

        for component in self.component_dict.keys():
            self.add_implicit_diffusion(component, kappa=theta, marker=0)
            self.add_explicit_diffusion(component, u, kappa=one-theta, marker=0)

        if self.is_same_diffusivity==False:
            self.add_semi_implicit_charge_balanced_diffusion(u, kappa=theta, marker=0)
            self.add_explicit_charge_balanced_diffusion(u, kappa=one-theta, marker=0)

        # self.evaluate_jacobian(self.get_forms()[0])

    def solve_solvent_transport(self):
        fluid_comp_old = self.fluid_components.x.array
        fluid_comp_new = self.get_solver_u1().x.array

        self.solvent.x.array[:] += \
        ((fluid_comp_old - np.exp(fluid_comp_new)).reshape(-1, self.num_component)\
        *self._M_fraction).sum(axis=1)

    def setup_transport_solver(self):
        self.set_advection_velocity()
        self.generate_solver(eval_jacobian=True)
        self.set_solver_parameters('gmres', 'jacobi')

    def set_solver_parameters(self, linear_solver='gmres', preconditioner='jacobi'):
        super().set_solver_parameters(linear_solver, preconditioner)
        self.set_krylov_solver_params(self.get_solver())

    def set_krylov_solver_params(self, prm):
        prm.convergence_criterion = "residual"
        prm.atol = 1e-13
        prm.rtol = 1e-12
        prm.max_it = 100
        prm.nonzero_initial_guess = True
        prm.report = True