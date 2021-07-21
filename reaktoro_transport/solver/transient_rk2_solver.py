from . import *

class TransientRK2Solver:
    """A solver class that is used as a mixin for problem classes."""

    num_forms = 2

    def generate_solver(self):

        self.__func_space = self.get_function_space()

        self.__u0 = self.fluid_components
        self.__u1 = Function(self.comp_func_spaces)
        self.kappa = Constant(0.5)
        one = Constant(1.0)

        self.add_physics_to_form(self.__u0, self.kappa, f_id=0)
        self.add_time_derivatives(self.__u0, f_id=0)

        self.add_physics_to_form(self.__u1, Constant(0.5)/self.kappa, f_id=1)
        self.add_physics_to_form(self.__u0, one - Constant(0.5)/self.kappa, f_id=1)
        self.add_time_derivatives(self.__u0, f_id=1)

        self.__forms = self.get_forms()
        self.__form1 = self.__forms[0]
        self.__form2 = self.__forms[1]

        a, L = lhs(self.__form1), rhs(self.__form1)

        problem1 = LinearVariationalProblem(a, L, self.__u1, self.get_dirichlet_bcs())
        problem2 = LinearVariationalProblem(a, L, self.__u0, self.get_dirichlet_bcs())

        self.__solver1 = LinearVariationalSolver(problem1)
        self.__solver2 = LinearVariationalSolver(problem2)

    def set_solver_parameters(self, linear_solver='gmres', preconditioner='amg'):
        prm = self.__solver1.parameters
        prm['linear_solver'] = linear_solver
        prm['preconditioner'] = preconditioner

        set_default_solver_parameters(prm['krylov_solver'])

        prm = self.__solver2.parameters
        prm['linear_solver'] = linear_solver
        prm['preconditioner'] = preconditioner

        set_default_solver_parameters(prm['krylov_solver'])

    def set_kappa(self, kappa):
        """
        When kappa=0.5, the RK2 method results in the midpoint scheme.
        When kappa=1.0, the RK2 method results in Heun's method. If both steps
        are TVD (Total Variation Diminishing), such Heun's method is also known
        as SSP (Strong Stability Preserving) methods.
        """

        self.kappa.assign(kappa)

    def set_dt(self, dt_val):
        self.dt.assign(dt_val)

    def solve_one_step(self):
        self.__solver1.solve()
        self.__solver2.solve()

    def solve_transport(self, dt_val=1.0, timesteps=1):
        """"""

        self.set_dt(dt_val)
        self.save_to_file(time=0.0)

        for i in range(timesteps):
            self.solve_one_step()
            self.save_to_file(time=(i+1)*dt_val)
