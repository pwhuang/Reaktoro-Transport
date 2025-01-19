# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os
os.environ['OMP_NUM_THREADS'] = '1'

from mesh_factory import MeshFactory
from retropy.problem import DarcyFlowMixedPoisson, TracerTransportProblem
from retropy.manager import ReactiveTransportManager, TransportManager
from retropy.physics import DG0Kernel
from retropy.manager import XDMFManager as OutputManager
from retropy.solver import PETScSolver

from dolfinx.fem import Constant, form, Function
from dolfinx.fem.petsc import assemble_vector
from ufl import as_vector

import reaktoro as rkt
import numpy as np

class FlowManager(DarcyFlowMixedPoisson):
    def setup_flow_solver(self):
        self.set_flow_fe_space()
        self.set_fluid_properties()

        self.generate_form()
        self.generate_residual_form()
        self.set_flow_ibc()

        # self.set_additional_parameters(r_val=0.0, omega_by_r=1.0)
        self.assemble_matrix()

        solver_params = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

        self.set_flow_solver_params(solver_params)

    def set_flow_fe_space(self):
        self.set_pressure_fe_space('DG', 0)

        if self.mesh.topology.cell_name() == 'triangle':
            self.set_velocity_fe_space('BDM', 1)
        elif self.mesh.topology.cell_name() == 'quadrilateral':
            self.set_velocity_fe_space('RTCF', 1)

    def set_flow_ibc(self):
        """Sets the initial and boundary conditions of the flow."""

        self.set_pressure_ic(lambda x: 0.0 * x[0] + 101325.)
        self.set_pressure_bc({'top': Constant(self.mesh, 101325. + 100.),
                              'bottom': Constant(self.mesh, 101325.),})
        
        # self.set_pressure_bc({'bottom': Constant(self.mesh, 101325.),})
        self.add_weak_pressure_bc(penalty_value=20.0)

        velocity_bc = Function(self.velocity_func_space)
        velocity_bc.interpolate(lambda x: (0.0*x[0], 0.0*x[1]))

        velocity_top = Function(self.velocity_func_space)
        velocity_top.interpolate(lambda x: (0.0*x[0], -1.0 + 0.0*x[1]))

        self.set_velocity_bc({'left': velocity_bc,
                              'right': velocity_bc,})

class ReactiveTransportManager(ReactiveTransportManager):
    def __init__(self, marked_mesh):
        super().__init__(marked_mesh)
        self.interpolated_velocity = Function(self.Vec_DG0_space)
        self.interpolated_velocity.name = 'fluid velocity'

    def add_physics_to_form(self, u, kappa, f_id=0):
        theta_val = 0.5

        theta = Constant(self.mesh, theta_val)
        one = Constant(self.mesh, 1.0)

        self.add_explicit_advection(u, kappa=one, marker=0, f_id=f_id)
        # self.add_implicit_advection(kappa=one, marker=0, f_id=f_id)

        for component in self.component_dict.keys():
            self.add_implicit_diffusion(component, kappa=theta, marker=0)
            self.add_explicit_diffusion(component, u, kappa=one-theta, marker=0)

        if self.is_same_diffusivity==False:
            self.add_semi_implicit_charge_balanced_diffusion(u, kappa=theta, marker=0)
            self.add_explicit_charge_balanced_diffusion(u, kappa=one-theta, marker=0)

    def set_chem_system(self, database):
        db = rkt.PhreeqcDatabase('pitzer.dat')
        aqueous_components = self.component_str + ' ' + self.solvent_name

        self.aqueous_phase = rkt.AqueousPhase(aqueous_components)
        self.mineral_phase = rkt.MineralPhase(self.mineral_name)
        self.chem_system = rkt.ChemicalSystem(db, self.aqueous_phase, self.mineral_phase)
        
        self.chem_sys_dof = self.chem_system.species().size()
        self.solvent_idx = self.chem_system.species().index(self.solvent_name)

    def set_activity_models(self):
        self.aqueous_phase.set(rkt.chain(rkt.ActivityModelPitzer()))

    def set_advection_velocity(self):
        self.advection_velocity = as_vector(
            [self.fluid_velocity / self._phi for _ in range(self.num_component)]
        )

    def setup_reaction_solver(self, temp=298.15):
        super().setup_reaction_solver()
        num_dof = self.get_num_dof_per_component()
        self.molar_density_temp = np.zeros([num_dof, self.num_component+2])

    def _solve_chem_equi_over_dofs(self, pressure, fluid_comp):
        for i in self.dof_idx:
            self._set_pressure(pressure[i], 'Pa')
            self._set_species_amount(list(fluid_comp[i]) + [self.solvent.x.array[i]] + [self.mineral.x.array[i]])
            self.solve_chemical_equilibrium()

            self.rho_temp[i] = self._get_fluid_density()
            self.pH_temp[i] = self._get_fluid_pH()
            self.molar_density_temp[i] = self._get_species_amounts()

    def _assign_chem_equi_results(self):
        super()._assign_chem_equi_results()
        self.mineral.x.array[:] = self.molar_density_temp[:, -1].flatten()

    def save_to_file(self, time):
        super().save_to_file(time)
        self.write_function(self.mineral, time)
        self.interpolated_velocity.interpolate(self.fluid_velocity)
        self.write_function(self.interpolated_velocity, time)
        self.write_function(self.fluid_pressure, time)

    @staticmethod
    def timestepper(dt_val, current_time, time_stamp):
        min_dt, max_dt = 5e-3, 10.0

        if (dt_val := dt_val*1.1) > max_dt:
            dt_val = max_dt
        elif dt_val < min_dt:
            dt_val = min_dt
        if dt_val > time_stamp - current_time:
            dt_val = time_stamp - current_time

        return dt_val
    
    def solve(self, dt_val=1.0, endtime=10.0, time_stamps=[]):
        current_time = 0.0
        timestep = 1
        saved_times = []
        flow_residuals = []
        self.trial_count = 0

        time_stamps.append(endtime)
        time_stamp_idx = 0
        time_stamp = time_stamps[time_stamp_idx]

        # self.solve_initial_condition()
        self.save_to_file(time=current_time)

        saved_times.append(current_time)
        flow_residuals.append(self.get_flow_residual())
        save_interval = 1
        flush_interval = 25

        while current_time < endtime:
            if self.__MPI_rank==0:
                print(f"timestep = {timestep}, dt = {dt_val:.6f}, "\
                      f"current_time = {current_time:.6f}\n")

            self.dt.value = dt_val

            newton_steps, is_solved = self.solve_species_transport()

            if is_solved is False:
                dt_val = 0.7*dt_val
                self.trial_count += 1
                continue

            if self.__MPI_rank==0:
                print(f"Transport solve converged. Newton steps = {newton_steps}.\n")

            self.trial_count = 0
            self.solve_solvent_transport()
            
            self._assign_chem_equi_results()
            self.solve_flow(target_residual=self.flow_residual, max_steps=20)

            timestep += 1

            current_time = current_time + dt_val
            if current_time >= time_stamp:
                time_stamp_idx += 1
                try:
                    time_stamp = time_stamps[time_stamp_idx]
                except:
                    time_stamp = time_stamps[-1]

            dt_val = self.timestepper(dt_val, current_time, time_stamp)

            if timestep % save_interval == 0:
                self.save_to_file(time=current_time)
                saved_times.append(current_time)
                flow_residuals.append(self.get_flow_residual())

            if timestep % flush_interval == 0:
                self.flush_output()

        if self.__MPI_rank==0:
            np.save(self.output_file_name + '_time', np.array(saved_times), allow_pickle=False)
            np.save(self.output_file_name + '_flow_res', np.array(flow_residuals), allow_pickle=False)

class TransportManager(TracerTransportProblem, DG0Kernel, PETScSolver):
    def __init__(self, marked_mesh):
        TracerTransportProblem.__init__(self, marked_mesh)

    def add_physics_to_form(self, u, kappa, f_id=0):
        self.add_explicit_advection(u, kappa=kappa, marker=0, f_id=f_id)
        
        for component in self.component_dict.keys():
            self.add_implicit_diffusion(component, kappa=kappa, marker=0)

    def setup_transport_solver(self):
        self.set_advection_velocity()
        self.generate_solver()
        self.set_solver_parameters('gmres', 'jacobi')

    def calculate_delta_S(self):
        S1 = self.get_solver_u1().x.array.reshape(-1, self.num_component).T[-1]
        S0 = self.fluid_components.x.array.reshape(-1, self.num_component).T[-1]
        self.delta_S = S1 - S0

    def solve_transport(self):
        self.solve_one_step()
        self.calculate_delta_S()
        self.assign_u1_to_u0()

class Problem(TransportManager, FlowManager, OutputManager):
    """This class solves the CO2 convection problem."""

    def __init__(self, nx, ny, const_diff):
        super().__init__(MeshFactory(nx, ny, mesh_type='triangle'))
        self.is_same_diffusivity = const_diff
        # self.set_flow_residual(5e-10)

    def set_component_properties(self):
        self.set_molar_mass([40., 60., 100.]) #g/mol
        self.set_solvent_molar_mass(18.0153)
        self.set_charge([2.0, -2.0, 0.0])

    def define_problem(self):
        self.set_components('Ca+2 CO3-2 Calcite')
        self.set_component_mobility([True, True, False])
        self.set_solvent('H2O')
        # self.set_mineral('Calcite')
        self.set_component_properties()
        # self.Mv_calcite = 37.65 * 1e-3 # mm3/umol
        self.Mv_calcite = 4.0

        self.set_component_fe_space()
        self.initialize_form()

        # self.background_pressure = 101325. # Pa, 1 atm
        self.injected_amount = Constant(self.mesh, 1.0)  # mol / L == umol / mm3
        init_conc = [1e-15, 1e-15, 1e-15, 55.0]

        for comp, concentration in zip(self.component_dict.keys(), init_conc):
            self.set_component_ics(comp, lambda x: 0.0 * x[0] + concentration)

        self.set_solvent_ic(lambda x: 0.0 * x[0] + init_conc[-1])
        # self.set_mineral_ic(lambda x: 0.0 * x[0] + 1.0)

        self.mark_component_boundary(
            {
                "Ca+2": [self.marker_dict["top"]],
                "CO3-2": [self.marker_dict["top"]],
                "outlet": [self.marker_dict["bottom"]],
            }
        )

    def kinetics(self, C1, C2):
        kd = Constant(self.mesh, 0.0)
        kp = Constant(self.mesh, 2e-3)

        return -kd + kp*C1*C2

    def add_physics_to_form(self, u, kappa, f_id=0):
        super().add_physics_to_form(u, kappa, f_id)
        one = Constant(self.mesh, 1.0)

        self.add_component_time_derivative(u, "Ca+2", kappa=self._phi)
        self.add_component_time_derivative(u, "CO3-2", kappa=self._phi)
        self.add_component_time_derivative(u, "Calcite", kappa=one)

        C1 = self.get_trial_function()[0]
        # C2 = self.get_trial_function()[1]
        C2 = u[1]

        self.add_mass_source(["Ca+2"], [-self.kinetics(C1, C2)], kappa, f_id)
        self.add_mass_source(["CO3-2"], [-self.kinetics(C1, C2)], kappa, f_id)
        self.add_mass_source(["Calcite"], [self.kinetics(C1, C2)], kappa, f_id)

        source = Constant(self.mesh, 0.05)
        self.add_mass_source(["CO3-2"], [source], kappa, f_id)

        inlet_conc = [self.injected_amount]  # micro mol/mm^3 # mol/L

        for comp, conc in zip(["Ca+2"], inlet_conc):
            self.add_component_advection_bc(comp, [conc], kappa, f_id)
        
        self.add_outflow_bc(f_id)

    def set_fluid_properties(self):
        self.set_porosity(1.0)
        self.set_fluid_density(1e-3) # Initialization # g/mm^3
        self.set_fluid_viscosity(0.893e-3)  # Pa sec
        self.set_gravity([0.0, 0.0]) # mm/sec^2
        
        self.hmax = Constant(self.mesh, 0.2) ## mm
        self.aperture_width = Function(self.DG0_space)
        self.aperture_width.name = 'aperture'
        
        self.aperture_width.interpolate(lambda x: self.hmax.value - 4e-4 * x[0])
        self.set_permeability(0.0) # mm^2
        self._k.x.array[:] = self.aperture_width.x.array[:]**2 / 12.
        self._k.x.scatter_forward()

    def update_permeability(self):
        # TODO: define permeability based on local cubic law or Kozeny-Carman equation
        self.aperture_width.x.array[:] -= self.hmax.value * self.Mv_calcite * self.delta_S

        # self.aperture_width.x.array[self.aperture_width.x.array < 0.0] = 1e-15
        
        self._k.x.array[:] = self.aperture_width.x.array[:]**2 / 12.
        self._k.x.scatter_forward()

    def save_to_file(self, time, is_saving_pv=False):
        super().save_to_file(time, is_saving_pv=False)
        self.write_function(self.fluid_pressure, time)
        self.write_function(self.aperture_width, time)
    
    def solve(self, dt_val=1.0, timesteps=1):
        self.dt.value = dt_val
        saved_times = []

        for i in range(timesteps):
            self.solve_flow(target_residual=5e-9, max_steps=50)
            self.solve_transport()
            self.update_permeability()

            if (i+1) * self.dt.value > 3.0:
                self.injected_amount.value = 0.0

            self.save_to_file(time=(i+1)*self.dt.value, is_saving_pv=False)
            saved_times.append((i+1)*self.dt.value)

        if self.mesh.comm.Get_rank()==0:
            np.save(self.output_file_name + '_time', np.array(saved_times), allow_pickle=False)