import sys
sys.path.insert(0, '../../')

from reaktoro_transport.problem import DarcyFlowUzawa
from dolfin import Constant, Function, info, PETScKrylovSolver
from numpy import abs, max

def set_krylov_solver_params(prm):
    prm['absolute_tolerance'] = 1e-10
    prm['relative_tolerance'] = 1e-14
    prm['maximum_iterations'] = 8000
    prm['error_on_nonconvergence'] = True
    prm['monitor_convergence'] = False
    prm['nonzero_initial_guess'] = False

class FlowManager(DarcyFlowUzawa):
    def set_fluid_properties(self):
        self.set_porosity(1.0)
        self.set_permeability(0.5**2/12.0) # mm^2
        self.set_fluid_density(1e-3) # Initialization # g/mm^3
        self.set_fluid_viscosity(8.9e-4)  # Pa sec
        self.set_gravity([0.0, -9806.65]) # mm/sec

    def setup_flow_solver(self):
        self.set_pressure_fe_space('DG', 0)
        self.set_velocity_fe_space('BDM', 1)

        self.set_pressure_ic(Constant(0.0))
        self._rho_old = Function(self.pressure_func_space)

        self.set_fluid_properties()

        self.mark_flow_boundary(pressure = [],
                                velocity = [self.marker_dict['top'], self.marker_dict['bottom'],
                                            self.marker_dict['left'], self.marker_dict['right']])

        self.set_pressure_bc([]) # Pa
        self.generate_form()
        self.generate_residual_form()
        #self.add_mass_source([-(self.fluid_density - self._rho_old)/self.dt])
        self.set_velocity_bc([Constant([0.0, 0.0])]*4)

        self.set_solver()
        #set_krylov_solver_params(prm)
        self.set_additional_parameters(r_val=2e4, omega_by_r=1.3)
        self.assemble_matrix()

    def solve_flow(self, target_residual: float, max_steps: int):
        super().solve_flow(target_residual, max_steps)

        #info('Max velocity: ' + str( (self.fluid_velocity.vector().max() )))

    def set_solver(self, **kwargs):
        # Users can override this method.
        # Or, TODO: make this method more user friendly.

        self.solver_v = PETScKrylovSolver('bicgstab', 'jacobi')
        self.solver_p = PETScKrylovSolver('gmres', 'none')

        prm_v = self.solver_v.parameters
        prm_p = self.solver_p.parameters

        set_krylov_solver_params(prm_v)
        set_krylov_solver_params(prm_p)