from . import *

class TracerTransportProblem(TransportProblemBase,
                             MassBalanceBase, ComponentProperty):
    """A class that solves single-phase tracer transport problems."""

    def __init__(self, mesh, boundary_markers, domain_markers):
        try:
            super().num_forms
        except:
            raise Exception("num_forms does not exist. Consider inherit a solver class.")

        self.set_mesh(mesh)
        self.set_boundary_markers(boundary_markers)
        self.set_domain_markers(domain_markers)

        self.dt = Constant(1.0)
        self.__dirichlet_bcs = []

    def mark_component_boundary(self, **kwargs):
        """This method gives boundary markers physical meaning.

        Keywords
        ---------
        {component_name : markers}
        Example: {'Na+': [1, 2, 3], 'outlet': [4]}
        """

        self.__boundary_dict = kwargs

    def set_component_fe_space(self):
        self.FiniteElement = FiniteElement(super().fe_space,
                                           self.mesh.ufl_cell(),
                                           super().fe_degree)

        element_list = []
        for i in range(self.num_component):
            element_list.append(self.FiniteElement)

        self.comp_func_spaces = FunctionSpace(self.mesh,
                                              MixedElement(element_list))

        #self.__function_space = FunctionSpace(self.mesh, self.FiniteElement)
        self.__function_space = FunctionSpace(self.mesh, super().fe_space,
                                              super().fe_degree)

        self.func_space_list = []

        if self.num_component==1:
            self.func_space_list.append(self.comp_func_spaces)

        else:
            for i in range(self.num_component):
                self.func_space_list.append(self.comp_func_spaces.sub(i).collapse())

        self.output_func_spaces = [self.__function_space]*self.num_component
        self.function_list = [Function(self.__function_space)]*self.num_component
        self.output_assigner = FunctionAssigner(self.output_func_spaces,
                                                self.comp_func_spaces)

    def get_function_space(self):
        return self.comp_func_spaces

    def get_fluid_components(self):
        return self.fluid_components

    def initialize_form(self):
        """"""

        self.__u = TrialFunction(self.comp_func_spaces)
        self.__w = TestFunction(self.comp_func_spaces)

        self.fluid_components = Function(self.comp_func_spaces)
        self.__u0 = self.fluid_components

        self.tracer_forms = [Constant(0.0)*inner(self.__w, self.__u)*self.dx]*super().num_forms

    def set_component_ics(self, expressions: Expression):
        """"""

        if len(expressions)!=self.num_component:
            raise Exception("length of expressions != num_components")

        # init_conds = []
        # for i, expression in enumerate(expressions):
        #     init_conds.append(interpolate(expression, self.func_space_list[i]))

        self.fluid_components.assign(interpolate(expressions, self.comp_func_spaces))

    def set_component_ic(self, component_name: str, expression):
        """"""
        #TODO: Make this function work.

        idx = self.component_dict[component_name]
        self.__u0[idx].assign(interpolate(expression, self.func_space_list[i]))

    def add_component_advection_bc(self, component_name: str, values, f_id=0):
        """"""

        if len(values)!=len(self.__boundary_dict[component_name]):
            raise Exception("length of values != number of markers")

        idx = self.component_dict[component_name]
        markers = self.__boundary_dict[component_name]

        for i, marker in enumerate(markers):
            self.tracer_forms[f_id] += self.advection_flux_bc(self.__w[idx], values[i], marker)

    def add_component_diffusion_bc(self, component_name: str, diffusivity, values, f_id=0):
        """"""

        if len(values)!=len(self.__boundary_dict[component_name]):
            raise Exception("length of values != number of markers")

        idx = self.component_dict[component_name]
        markers = self.__boundary_dict[component_name]

        for i, marker in enumerate(markers):
            self.tracer_forms[f_id] += self.diffusion_flux_bc(self.__w[idx], self.__u[idx],
                                                              diffusivity, values[i], marker)

    def add_component_dirichlet_bc(self, component_name: str, values):
        """"""

        if len(values)!=len(self.__boundary_dict[component_name]):
            raise Exception("length of values != number of markers")

        idx = self.component_dict[component_name]
        markers = self.__boundary_dict[component_name]

        for i, marker in enumerate(markers):
            bc = DirichletBC(self.func_space_list[idx], [values[i], ],
                             self.boundary_markers, marker)
            self.__dirichlet_bcs.append(bc)

    def add_outflow_bc(self, f_id=0):
        """"""

        for i, marker in enumerate(self.__boundary_dict['outlet']):
            self.tracer_forms[f_id] += self.advection_outflow_bc(self.__w, self.__u, marker)

    def add_time_derivatives(self, u, kappa=Constant(1.0), f_id=0):
        self.tracer_forms[f_id] += kappa*self.d_dt(self.__w, self.__u, u)

    def add_explicit_advection(self, u, kappa=Constant(1.0), marker=0, f_id=0):
        """Adds explicit advection physics to the variational form."""

        self.tracer_forms[f_id] += kappa*self.advection(self.__w, u, marker)

    def add_implicit_advection(self, kappa=Constant(1.0), marker=0, f_id=0):
        """Adds implicit advection physics to the variational form."""

        self.tracer_forms[f_id] += kappa*self.advection(self.__w, self.__u, marker)

    def add_explicit_diffusion(self, component_name: str, u, kappa=Constant(1.0), marker=0, f_id=0):
        """Adds explicit diffusion physics to the variational form."""

        idx = self.component_dict[component_name]
        self.tracer_forms[f_id] += kappa*self.diffusion(self.__w[idx], u[idx], self._D[idx], marker)

    def add_implicit_diffusion(self, component_name: str, kappa=Constant(1.0), marker=0, f_id=0):
        """Adds implicit diffusion physics to the variational form."""

        idx = self.component_dict[component_name]
        self.tracer_forms[f_id] += kappa*self.diffusion(self.__w[idx], self.__u[idx], self._D[idx], marker)

    def add_dispersion(self):
        return #TODO: Setup this method.

    def add_mass_source(self, component_names: list[str], sources: list, f_id=0):
        """Adds mass source to the variational form."""

        for i, component_name in enumerate(component_names):
            idx = self.component_dict[component_name]
            self.tracer_forms[f_id] -= self.__w[idx]*sources[i]*self.dx

    def get_forms(self):
        return self.tracer_forms

    def get_dirichlet_bcs(self):
        return self.__dirichlet_bcs

    def save_to_file(self, time: float):
        """"""

        try:
            self.xdmf_obj
        except:
            return False

        is_appending = True

        if self.num_component==1:
            self.output_assigner.assign(self.function_list[0], self.fluid_components)
        else:
            self.output_assigner.assign(self.function_list, self.fluid_components)

        for key, i in self.component_dict.items():
            self.xdmf_obj.write_checkpoint(self.function_list[i], key,
                                           time_step=time,
                                           append=is_appending)

        return True
