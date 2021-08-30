import sys
sys.path.insert(0, '../../')

from reaktoro_transport.problem import DarcyFlowUzawa
from reaktoro_transport.tests import convergence_rate
from reaktoro_transport.tests.benchmarks import DarcyMassSourceBenchmark

from math import isclose

class DarcyMassMixedPoissonTest(DarcyFlowUzawa, DarcyMassSourceBenchmark):
    """"""

    def __init__(self, nx):
        DarcyFlowUzawa.__init__(self, *self.get_mesh_and_markers(nx))

        self.set_pressure_fe_space('DG', 0)
        self.set_velocity_fe_space('BDM', 1)
        self.get_solution()

        self.set_material_properties()
        self.set_boundary_conditions()
        self.set_mass_sources()

        self.set_solver()
        self.set_additional_parameters(r_val=1.0, omega_by_r=1.1)
        self.assemble_matrix()

# nx is the mesh element in one direction.
list_of_nx = [10, 15]
element_diameters = []
p_err_norms = []
v_err_norms = []

for nx in list_of_nx:
    problem = DarcyMassMixedPoissonTest(nx)
    problem.solve_flow(target_residual=1e-10, max_steps=10)
    pressure_error_norm, velocity_error_norm = problem.get_error_norm()

    p_err_norms.append(pressure_error_norm)
    v_err_norms.append(velocity_error_norm)
    element_diameters.append(problem.get_mesh_characterisitic_length())

    print(problem.get_residual())

convergence_rate_p = convergence_rate(p_err_norms, element_diameters)
convergence_rate_v = convergence_rate(v_err_norms, element_diameters)

print(convergence_rate_p, convergence_rate_v)

def test_function():
    assert isclose(convergence_rate_p, 2, rel_tol=0.05)\
       and isclose(convergence_rate_v, 2, rel_tol=0.05)