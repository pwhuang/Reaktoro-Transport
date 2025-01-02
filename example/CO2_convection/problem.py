# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os
os.environ['OMP_NUM_THREADS'] = '1'

from mesh_factory import MeshFactory
from retropy.manager import DarcyFlowManagerUzawa as FlowManager
from retropy.manager import ReactiveTransportManager
from retropy.manager import XDMFManager as OutputManager
from retropy.solver import TransientNLSolver

from retropy.problem import MassBalanceBase
from retropy.manager import ReactionManager

from dolfinx.fem import Constant, form
from dolfinx.fem.petsc import assemble_vector
from ufl import TestFunction, FacetArea

import reaktoro as rkt
import numpy as np

class BoundaryEquilibriumProblem(MassBalanceBase, ReactionManager):
    def __init__(self, component, solvent):
        self.set_components(component)
        self.set_solvent(solvent)
        self.H_idx = self.component_dict['H+']

    def set_chem_system(self, database="supcrt07"):
        db = rkt.SupcrtDatabase(database)
        aqueous_components = self.component_str + ' ' + self.solvent_name

        self.aqueous_phase = rkt.AqueousPhase(aqueous_components)
        self.gaseous_phase = rkt.GaseousPhase('CO2(g)')
        self.chem_system = rkt.ChemicalSystem(db, self.aqueous_phase, self.gaseous_phase)

    def set_activity_models(self):
        self.aqueous_phase.set(rkt.chain(rkt.ActivityModelHKF(),
                                         rkt.ActivityModelDrummond("CO2")))
        self.gaseous_phase.set(rkt.ActivityModelPengRobinson())

class ReactiveTransportManager(ReactiveTransportManager):
    def setup_auxiliary_reaction_solver(self):
        self.total_gaseous_CO2_amount = 1e5 # mols
        self.aux_equi_problem = BoundaryEquilibriumProblem(self.component_str,
                                                           self.solvent_name)

        self.aux_equi_problem.initialize_Reaktoro()
        self.aux_equi_problem._set_temperature(298.15, 'K')
        self.aux_equi_problem._set_pressure(1e5, 'Pa')

    def mark_inflow_boundary_cells(self):
        w = TestFunction(self.DG0_space)

        one = Constant(self.mesh, 1.0)

        cell_marker = assemble_vector(form(w * one * self.ds(self.marker_dict['top']))).array[:]
        cell_marker = np.array(cell_marker, dtype=float)

        idx = np.arange(0, cell_marker.size)

        boundary_cell_idx = idx[cell_marker > 1e-10]
        domain_cell_idx = idx[cell_marker < 1e-10]

        return boundary_cell_idx, domain_cell_idx

    def set_dof_idx(self):
        self.boundary_cell_idx, self.dof_idx = self.mark_inflow_boundary_cells()

    def set_activity_models(self):
        self.aqueous_phase.set(rkt.chain(rkt.ActivityModelHKF(),
                                         rkt.ActivityModelDrummond("CO2")))

    def _solve_chem_equi_over_dofs(self, pressure, fluid_comp):
        super()._solve_chem_equi_over_dofs(pressure, fluid_comp)

        for i in self.boundary_cell_idx:
            self.aux_equi_problem._set_species_amount(list(fluid_comp[i]) + \
                                                      [self.solvent.x.array[i]] + \
                                                      [self.total_gaseous_CO2_amount])

            self.aux_equi_problem.solve_chemical_equilibrium()

            self.rho_temp[i] = self.aux_equi_problem._get_fluid_density()
            self.pH_temp[i] = self.aux_equi_problem._get_fluid_pH()
            self.molar_density_temp[i] = self.aux_equi_problem._get_species_amounts()[:-1]

class FlowManager(FlowManager):
    def set_flow_fe_space(self):
        self.set_pressure_fe_space('DG', 0)
        self.set_velocity_fe_space('RTCF', 1)

class Problem(ReactiveTransportManager, FlowManager, OutputManager,
              TransientNLSolver):
    """This class solves the CO2 convection problem."""

    def __init__(self, nx, ny, const_diff):
        super().__init__(MeshFactory(nx, ny))
        self.is_same_diffusivity = const_diff
        self.set_flow_residual(5e-10)

    def set_component_properties(self):
        self.set_molar_mass([6.941, 1.00794, 17.00734, 44.0095, 60.0089, 61.01684]) #g/mol
        self.set_solvent_molar_mass(18.0153)
        self.set_charge([1.0, 1.0, -1.0, 0.0, -2.0, -1.0])

    def define_problem(self):
        self.set_components('Li+ H+ OH- CO2(aq) CO3-2 HCO3-')
        self.set_solvent('H2O(aq)')
        self.set_component_properties()

        self.set_component_fe_space()
        self.initialize_form()

        self.background_pressure = 1e5 + 1e-3*9806.65*20.0 # Pa

        LiOH_amounts = [0.01, 1e-15, 0.01, 1e-15, 1e-15, 1e-15, 55.345] # micro mol/mm^3 # mol/L

        for comp, concentration in zip(self.component_dict.keys(), LiOH_amounts):
            self.set_component_ics(comp, lambda x: 0.0 * x[0] + concentration)

        self.set_solvent_ic(lambda x: 0.0 * x[0] + LiOH_amounts[-1])

    def set_fluid_properties(self):
        self.set_porosity(1.0)
        self.set_fluid_density(1e-3) # Initialization # g/mm^3
        self.set_fluid_viscosity(0.893e-3)  # Pa sec
        self.set_gravity([0.0, -9806.65]) # mm/sec
        self.set_permeability(0.5**2/12.0) # mm^2

    @staticmethod
    def timestepper(dt_val, current_time, time_stamp):
        min_dt, max_dt = 5e-3, 2.0

        if (dt_val := dt_val*1.1) > max_dt:
            dt_val = max_dt
        elif dt_val < min_dt:
            dt_val = min_dt
        if dt_val > time_stamp - current_time:
            dt_val = time_stamp - current_time

        return dt_val
