# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import *

class DarcyFlowMixedPoisson(TransportProblemBase, DarcyFlowBase):
    """This class utilizes the mixed Poisson method to solve
    the pressure and velocity of Darcy's flow.
    """

    def generate_form(self):
        """Sets up the FeNiCs form of Darcy flow"""

        self.func_space_list = [self.velocity_finite_element,
                                self.pressure_finite_element]

        self.mixed_func_space = FunctionSpace(self.mesh,
                                              MixedElement(self.func_space_list))

        W = self.mixed_func_space

        (self.__u, self.__p) = TrialFunctions(W)
        (self.__v, self.__q) = TestFunctions(W)

        u, p = self.__u, self.__p
        v, q = self.__v, self.__q

        mu, k, rho, g, phi = self._mu, self._k, self._rho, self._g, self._phi
        dx, ds = self.dx, self.ds
        n = self.n

        self.__r = Constant(self.mesh, 0.0)
        r = self.__r

        self.mixed_form = mu/k*inner(v, u)*dx - inner(div(v), p)*dx \
                          + r*inner(div(v), div(phi*rho*u))*dx \
                          - inner(v, rho*g)*dx \
                          + q*div(phi*rho*u)*dx

        for i, key in enumerate(self.darcyflow_boundary_dict['pressure']):
            marker = self.marker_dict[key]
            self.mixed_form +=  self.pressure_bc[i] * inner(n, v) * ds(marker)

        self.functions_to_save = [self.fluid_pressure, self.fluid_velocity]

    def add_mass_source(self, sources):
        q, v, r = self.__q, self.__v, self.__r
        dx = self.dx

        for source in sources:
            self.mixed_form -= q*source*dx + r*inner(div(v), source)*dx

    def add_momentum_source(self, sources: list):
        v  = self.__v

        for source in sources:
            self.mixed_form -= inner(v, source)*self.dx

    def set_velocity_bc(self, velocity_bc_val: list):
        """"""

        DarcyFlowBase.set_velocity_bc(self, velocity_bc_val)
        self.mixed_velocity_bc = []
        
        for i, key in enumerate(self.darcyflow_boundary_dict['velocity']):
            dofs = locate_dofs_topological(V = (self.mixed_func_space.sub(0), self.velocity_func_space), 
                                           entity_dim = self.mesh.topology.dim - 1,
                                           entities = self.facet_dict[key])
            bc = dirichletbc(value = velocity_bc_val[i], dofs = dofs, 
                             V = self.mixed_func_space.sub(0))
            self.mixed_velocity_bc.append(bc)

    def set_additional_parameters(self, r_val: float, **kwargs):
        self.__r.value = r_val

    def assemble_matrix(self):
        self.__a, self.__L = lhs(self.mixed_form), rhs(self.mixed_form)

    def set_flow_solver_params(self, petsc_options):
        self.problem = LinearProblem(self.__a, self.__L, bcs=self.mixed_velocity_bc,
                                     petsc_options=petsc_options)

    def solve_flow(self, **kwargs):
        U = self.problem.solve()

        u, p = U.sub(0).collapse(), U.sub(1).collapse()
 
        self.fluid_velocity.x.array[:] = u.x.array
        self.fluid_pressure.x.array[:] = p.x.array

        self.fluid_velocity.x.scatter_forward()
        self.fluid_pressure.x.scatter_forward()
