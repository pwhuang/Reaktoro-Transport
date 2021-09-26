import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(0, '../../')

from mesh_factory import MeshFactory
from flow_manager_uzawa import FlowManager
from transport_manager_exp import TransportManager
from reaktoro_transport.tests import quick_plot

import numpy as np
from dolfin import info, DOLFIN_EPS, assemble, exp, begin, end

class main(FlowManager, TransportManager, MeshFactory):
    """
    This class solves the chemically driven convection problem.
    """

    def __init__(self, nx, ny):
        TransportManager.__init__(self, *self.get_mesh_and_markers(nx, ny))

    def _solve_chem_equi_over_dofs(self):
        pressure = self.fluid_pressure.vector()[:]
        component_molar_density = np.exp(self.fluid_components.vector()[:].reshape(-1, self.num_component))

        molar_density_list = []
        rho_list = []

        for i in range(self.num_dof):
            self._set_pressure(pressure[i] + self.background_pressure, 'Pa')
            #self._set_pressure(pressure[i], 'Pa')
            self._set_species_amount(list(component_molar_density[i]) + [self.solvent.vector()[i]])
            self.solve_chemical_equilibrium()
            self.molar_density_temp[i] = self._get_species_amounts()
            self.rho_temp[i] = self._get_fluid_density()*1e-6

        #print(self._get_species_amounts())
        #print(self.molar_density_temp[:, :-1].flatten())
        self.fluid_components.vector()[:] = np.log(self.molar_density_temp[:, :-1].flatten())
        #print(np.exp(self.fluid_components.vector()[:]))
        self.solvent.vector()[:] = self.molar_density_temp[:,-1].flatten()
        self.fluid_density.vector()[:] = self.rho_temp.flatten()

    def solve_solvent_amount(self, fluid_comp_new):
        self.solvent.vector()[:] =\
        self.solvent.vector()[:] + \
        ((np.exp(self.fluid_components.vector()[:]) - np.exp(fluid_comp_new.vector()[:])).reshape(-1, self.num_component)\
         *self._M_fraction).sum(axis=1)

    def solve_initial_condition(self):
        self._solve_chem_equi_over_dofs()
        self._rho_old.assign(self.fluid_density)

    def save_fluid_density(self, time):
        self.xdmf_obj.write_checkpoint(self.fluid_density,
                                       self.fluid_density.name(),
                                       time_step=time,
                                       append=True)

    def save_to_file(self, time):
        super().save_to_file(time, is_saving_pv=True)
        self.save_fluid_density(time)

    def timestepper(self):
        pass

    def solve(self, dt_val=1.0, endtime=10.0):
        self.solve_initial_condition()
        self.solve_flow(target_residual=5e-9, max_steps=100)

        current_time = 0.0
        min_dt = 1e-5
        max_dt = 1.0
        timestep = 1
        self.save_to_file(time=current_time)

        while current_time < endtime:
            info('timestep = ' + str(timestep) + ',  dt = ' + str(dt_val)\
                 + ', current_time = ' + str(current_time) )

            self.set_dt(dt_val)

            try:
                self.solve_one_step()
            except:
                self.assign_u0_to_u1()
                if (dt_val := 0.75*dt_val) < min_dt:
                    print('Reached minimum dt. Abort!')
                    break
                continue

            self.solve_solvent_amount(self.get_solution())
            self.assign_u1_to_u0()
            self._solve_chem_equi_over_dofs()

            self.solve_flow(target_residual=5e-9, max_steps=50)

            self._rho_old.assign(self.fluid_density)

            timestep += 1
            current_time += dt_val

            if (dt_val := dt_val*1.1) > max_dt:
                dt_val = max_dt

            if timestep%1 == 0:
                self.save_to_file(time=current_time)



problem = main(nx=16, ny=26)
problem.generate_output_instance(sys.argv[1])
problem.define_problem()
problem.setup_flow_solver()
problem.setup_transport_solver()
problem.solve(dt_val=1e-2, endtime=250.0)

#quick_plot(problem.fluid_components.sub(1), problem.fluid_components.sub(2))